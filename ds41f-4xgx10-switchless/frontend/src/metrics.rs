//! Fixed-cardinality numeric mirrors. The listener is an OS thread, not a Tokio task.
//! It never touches a Gate, SessionTable, EngineClient mutex, filesystem, or logs.
use std::fmt::Write as _;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(not(target_has_atomic = "64"))]
compile_error!("native telemetry requires lock-free 64-bit atomics");
pub const BOUNDS: [u64; 19] = [
    1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000,
    10000000, 20000000, 30000000, 60000000, 120000000, 300000000, 600000000,
];
pub struct Histogram {
    bins: [AtomicU64; 20],
    sum: AtomicU64,
}
impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}
impl Histogram {
    const fn new() -> Self {
        Self {
            bins: [const { AtomicU64::new(0) }; 20],
            sum: AtomicU64::new(0),
        }
    }
    pub fn observe(&self, d: Duration) {
        let us = d.as_micros().min(u64::MAX as u128) as u64;
        let b = BOUNDS.iter().position(|&x| us <= x).unwrap_or(19);
        self.sum.fetch_add(us, Relaxed);
        self.bins[b].fetch_add(1, Relaxed);
    }
    fn render(&self, out: &mut String, name: &str, labels: &str, help: &str) {
        writeln!(
            out,
            "# HELP pico:{name} {help}\n# TYPE pico:{name} histogram"
        )
        .unwrap();
        let mut count = 0;
        for (i, b) in self.bins.iter().enumerate() {
            count += b.load(Relaxed);
            let le = BOUNDS
                .get(i)
                .map_or_else(|| "+Inf".into(), |v| (*v as f64 / 1e6).to_string());
            writeln!(out, "pico:{name}_bucket{{{labels},le=\"{le}\"}} {count}").unwrap();
        }
        writeln!(
            out,
            "pico:{name}_count{{{labels}}} {count}\npico:{name}_sum{{{labels}}} {}",
            self.sum.load(Relaxed) as f64 / 1e6
        )
        .unwrap();
    }
}
pub struct Metrics {
    pub initialization: AtomicU64,
    pub connected: AtomicU64,
    pub admission: AtomicU64,
    pub running: AtomicU64,
    pub waiting: AtomicU64,
    pub pending: AtomicU64,
    pub sessions: AtomicU64,
    pub retained: AtomicU64,
    pub token_budget: AtomicU64,
    pub received: AtomicU64,
    pub http_status: [AtomicU64; 5],
    pub engine_done: [AtomicU64; 6],
    pub generation_tokens: AtomicU64,
    pub connection_errors: AtomicU64,
    pub queue: Histogram,
    pub ttft: Histogram,
}
const _: () = assert!(std::mem::size_of::<Metrics>() < (1 << 20));
pub static M: Metrics = Metrics {
    initialization: AtomicU64::new(0),
    connected: AtomicU64::new(0),
    admission: AtomicU64::new(1),
    running: AtomicU64::new(0),
    waiting: AtomicU64::new(0),
    pending: AtomicU64::new(0),
    sessions: AtomicU64::new(0),
    retained: AtomicU64::new(0),
    token_budget: AtomicU64::new(0),
    received: AtomicU64::new(0),
    http_status: [const { AtomicU64::new(0) }; 5],
    engine_done: [const { AtomicU64::new(0) }; 6],
    generation_tokens: AtomicU64::new(0),
    connection_errors: AtomicU64::new(0),
    queue: Histogram::new(),
    ttft: Histogram::new(),
};
pub fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
fn render(model: &str, started: u64, heartbeat: u64) -> String {
    let labels = format!("model_name=\"{}\"", escape(model));
    let mut o = String::with_capacity(16384);
    let mut scalar = |name: &str, kind: &str, help: &str, value: f64| {
        writeln!(
            o,
            "# HELP pico:{name} {help}\n# TYPE pico:{name} {kind}\npico:{name}{{{labels}}} {value}"
        )
        .unwrap();
    };
    scalar(
        "frontend_metrics_heartbeat_timestamp_seconds",
        "gauge",
        "Independent endpoint loop heartbeat; not inference health.",
        heartbeat as f64 / 1e6,
    );
    scalar(
        "frontend_process_start_time_seconds",
        "gauge",
        "Frontend metrics initialization time.",
        started as f64 / 1e6,
    );
    for (n, h, v) in [
        (
            "frontend_initialization_state",
            "0 initializing; 1 HTTP bound; 2 stopping.",
            &M.initialization,
        ),
        (
            "frontend_engine_connected",
            "Engine socket handshake complete.",
            &M.connected,
        ),
        (
            "frontend_admission_open",
            "Gate not held or closing; connectivity separately reported.",
            &M.admission,
        ),
        (
            "frontend_requests_running",
            "Requests holding a gate pass, including stream lifetime.",
            &M.running,
        ),
        (
            "frontend_requests_waiting",
            "Requests waiting at the admission gate.",
            &M.waiting,
        ),
        (
            "frontend_engine_pending",
            "Socket submits awaiting a terminal event.",
            &M.pending,
        ),
        (
            "frontend_sessions",
            "Retained conversation table entries, not core KV slots.",
            &M.sessions,
        ),
        (
            "frontend_session_retained_tokens",
            "Actual retained history token IDs, not active KV.",
            &M.retained,
        ),
        (
            "frontend_session_token_budget",
            "Configured retention budget; busy sessions may temporarily exceed it.",
            &M.token_budget,
        ),
    ] {
        scalar(n, "gauge", h, v.load(Relaxed) as f64);
    }
    scalar(
        "frontend_requests_received_total",
        "counter",
        "Chat handler entries, before authentication / JSON validation.",
        M.received.load(Relaxed) as f64,
    );
    scalar(
        "frontend_generation_tokens_total",
        "counter",
        "Engine tokens received on the socket; not filtered visible output.",
        M.generation_tokens.load(Relaxed) as f64,
    );
    scalar(
        "frontend_engine_connection_errors_total",
        "counter",
        "Socket connect / connection failures.",
        M.connection_errors.load(Relaxed) as f64,
    );
    writeln!(o,"# HELP pico:frontend_http_responses_total Chat response headers by status class; SSE errors may follow a 2xx header.\n# TYPE pico:frontend_http_responses_total counter").unwrap();
    for (i, v) in M.http_status.iter().enumerate() {
        writeln!(
            o,
            "pico:frontend_http_responses_total{{{labels},status_class=\"{}xx\"}} {}",
            i + 1,
            v.load(Relaxed)
        )
        .unwrap();
    }
    writeln!(o,"# HELP pico:frontend_engine_finished_total Terminal engine events, including synthesized disconnect errors.\n# TYPE pico:frontend_engine_finished_total counter").unwrap();
    for (n, v) in ["stop", "length", "context", "cancelled", "error", "unknown"]
        .iter()
        .zip(&M.engine_done)
    {
        writeln!(
            o,
            "pico:frontend_engine_finished_total{{{labels},finished_reason=\"{n}\"}} {}",
            v.load(Relaxed)
        )
        .unwrap();
    }
    M.queue.render(
        &mut o,
        "frontend_request_queue_time_seconds",
        &labels,
        "Admission gate enter to pass; successful passes only.",
    );
    M.ttft.render(
        &mut o,
        "frontend_time_to_first_engine_token_seconds",
        &labels,
        "Gen start to first engine event; excludes HTTP preparation and gate wait.",
    );
    o
}
// Linux CPU mask, called once in the metrics thread. No serving-thread affinity is changed.
fn pin_housekeeping() -> io::Result<()> {
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; 16];
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let cpu = (0..1024)
        .find(|&i| {
            !(5..=9).contains(&i) && i != 12 && i != 15 && mask[i / 64] & (1u64 << (i % 64)) != 0
        })
        .ok_or_else(|| {
            io::Error::other("no metrics housekeeping CPU distinct from reserved 5-9 / 12 / 15")
        })?;
    mask.fill(0);
    mask[cpu / 64] = 1u64 << (cpu % 64);
    if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
pub struct Endpoint {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Endpoint {
    /// Opt-in listener. Explicit address only, no wildcard, no implicit credential exposure.
    pub fn start(addr: SocketAddr, model: String) -> io::Result<Self> {
        if addr.ip().is_unspecified() || model.len() > 256 {
            return Err(io::Error::other(
                "metrics needs explicit IP and model name <=256 bytes",
            ));
        }
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        // Handshake makes affinity failure a startup error, not an unnoticed degraded endpoint.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let thread=std::thread::Builder::new().name("pico-metrics".into()).spawn(move || {
            if let Err(e)=pin_housekeeping() {let _=tx.send(Err(e)); return;}
            let _=tx.send(Ok(()));
            let started=now_us();
            struct Client {s:TcpStream, since:Instant, input:Vec<u8>, output:Vec<u8>, sent:usize}
            let mut clients:Vec<Client>=Vec::with_capacity(4);
            while !signal.load(Relaxed) {
                let heartbeat=now_us();
                for _ in 0..4 {
                    match listener.accept() {
                        Ok((s,_)) if clients.len()<4 => {
                            if s.set_nonblocking(true).is_ok() {clients.push(Client{s,since:Instant::now(),input:Vec::with_capacity(4096),output:Vec::new(),sent:0});}
                        }
                        Ok(_)=>{}, Err(e) if e.kind()==io::ErrorKind::WouldBlock=>break, Err(_)=>break,
                    }
                }
                clients.retain_mut(|c| {
                    if c.since.elapsed()>=Duration::from_secs(1) {return false;}
                    if c.output.is_empty() {
                        let mut b=[0u8;1024];
                        match c.s.read(&mut b) {
                            Ok(0)=>return false,
                            Ok(n)=>c.input.extend_from_slice(&b[..n]),
                            Err(e) if e.kind()==io::ErrorKind::WouldBlock=>return true,
                            Err(_)=>return false,
                        }
                        if c.input.len()>4096 {return false;}
                        if !c.input.windows(4).any(|x|x==b"\r\n\r\n") {return true;}
                        let metrics=c.input.starts_with(b"GET /metrics HTTP/1.");
                        let live=c.input.starts_with(b"GET /live HTTP/1.");
                        let body=if metrics {render(&model,started,heartbeat)} else if live {"ok\n".into()} else {"not found\n".into()};
                        if body.len()>(256<<10) {return false;}
                        let status=if metrics||live {"200 OK"} else {"404 Not Found"};
                        c.output=format!("HTTP/1.1 {status}\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",body.len()).into_bytes();
                    }
                    match c.s.write(&c.output[c.sent..]) {
                        Ok(0)=>false, Ok(n)=>{c.sent+=n;c.sent<c.output.len()},
                        Err(e)=>e.kind()==io::ErrorKind::WouldBlock,
                    }
                });
                std::thread::sleep(Duration::from_millis(5));
            }
        })?;
        if let Err(e) = rx
            .recv()
            .map_err(|_| io::Error::other("metrics initialization failed"))?
        {
            let _ = thread.join();
            return Err(e);
        }
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}
impl Drop for Endpoint {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn histogram_boundaries_and_labels() {
        let h = Histogram::new();
        h.observe(Duration::ZERO);
        h.observe(Duration::from_millis(1));
        h.observe(Duration::from_secs(900));
        let mut s = String::new();
        h.render(&mut s, "test_seconds", "model_name=\"test\"", "test");
        assert!(s.contains("le=\"0.001\"} 2"));
        assert!(s.contains("le=\"+Inf\"} 3"));
        assert!(s.contains("_count{model_name=\"test\"} 3"));
        let s = render("a\"b\\c\nd", 1, 2);
        assert!(s.contains("model_name=\"a\\\"b\\\\c\\nd\""));
        for line in s.lines().filter(|l| !l.starts_with('#')) {
            assert!(line.contains("model_name="));
        }
        assert!(s.len() < 256 << 10);
    }
    #[test]
    fn independent_thread_serves_while_runtime_absent() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = l.local_addr().unwrap();
        drop(l);
        let _ep = Endpoint::start(a, "test".into()).unwrap();
        // No Tokio runtime or engine. A slow partial scrape cannot block a second client.
        let mut slow = TcpStream::connect(a).unwrap();
        slow.write_all(b"GET /").unwrap();
        let mut s = TcpStream::connect(a).unwrap();
        s.set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        s.write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut text = String::new();
        s.read_to_string(&mut text).unwrap();
        assert!(text.starts_with("HTTP/1.1 200"));
        assert!(text.contains("frontend_metrics_heartbeat"));
    }
}
