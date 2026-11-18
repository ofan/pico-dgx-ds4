//! The gate in front of the engine. A request passes while the engine is connected and the ring is
//! not lent; otherwise it waits, up to the hold limit, and is then answered 503 with Retry-After.
//!
//! service.sh lends the ring to a development job through the admin socket (docs/engine.md, "Launch
//! and stop"): `hold IDLE_S` succeeds only when no request is in flight or waiting and none has come
//! for IDLE_S seconds, and from then on requests wait here; the engine stops, the job runs, and
//! `release` once the ring is back lets them through as soon as the engine is connected again. A
//! request in flight is never cut: the hold waits for it.

use crate::engine::EngineClient;
use crate::protocol::SteerState;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

/// What a request answered 503 is told to wait (Retry-After, seconds).
pub const RETRY_AFTER_S: u64 = 60;
const POLL: Duration = Duration::from_millis(200);

struct State {
    /// requests wait: the ring is lent (since then)
    held: Option<Instant>,
    /// requests wait: a steering change is draining and applying (docs/steering-control.md)
    steering: bool,
    /// the server is shutting down: waiting requests are answered at once
    closing: bool,
    in_flight: u32,
    waiting: u32,
    /// the last request's arrival, start or end, whether it passed or not
    last: Instant,
    /// since the start: requests that waited and passed, and those answered 503
    passed: u64,
    refused: u64,
}

pub struct Gate {
    st: Mutex<State>,
    hold: Duration,
}

/// A request in flight: dropping it counts the request done.
pub struct Pass {
    gate: Arc<Gate>,
}

impl Drop for Pass {
    fn drop(&mut self) {
        let mut s = self.gate.st.lock().unwrap();
        s.in_flight -= 1;
        s.last = Instant::now();
    }
}

/// Requests are held off while a steering change drains and applies; dropping it admits them again.
pub struct Quiesced {
    gate: Arc<Gate>,
}

impl Drop for Quiesced {
    fn drop(&mut self) {
        self.gate.st.lock().unwrap().steering = false;
    }
}

/// A request waiting at the gate; counted out when it passes, is refused, or its client goes away.
struct Waiting {
    gate: Arc<Gate>,
    counted: bool,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.counted {
            // a request that waited and left (refused, or its client gone) was activity too
            let mut s = self.gate.st.lock().unwrap();
            s.waiting -= 1;
            s.last = Instant::now();
        }
    }
}

/// Why a request did not pass.
#[derive(Debug, PartialEq)]
pub enum Refused {
    /// it waited the hold limit
    Held(Duration),
    /// the server is shutting down
    Closing,
}

impl Gate {
    pub fn new(hold: Duration) -> Arc<Gate> {
        Arc::new(Gate {
            st: Mutex::new(State {
                held: None,
                steering: false,
                closing: false,
                in_flight: 0,
                waiting: 0,
                last: Instant::now(),
                passed: 0,
                refused: 0,
            }),
            hold,
        })
    }

    pub fn hold_limit(&self) -> Duration {
        self.hold
    }

    /// Wait until the request may go to the engine: the ring is not lent and the engine is connected.
    pub async fn enter(self: &Arc<Gate>, engine: &EngineClient) -> Result<Pass, Refused> {
        self.enter_when(|| engine.info().is_some()).await
    }

    async fn enter_when(self: &Arc<Gate>, connected: impl Fn() -> bool) -> Result<Pass, Refused> {
        let t0 = Instant::now();
        let mut w = Waiting { gate: self.clone(), counted: false };
        loop {
            {
                let mut s = self.st.lock().unwrap();
                if s.closing {
                    return Err(Refused::Closing);
                }
                if s.held.is_none() && !s.steering && connected() {
                    s.in_flight += 1;
                    s.last = Instant::now();
                    if w.counted {
                        w.counted = false;
                        s.waiting -= 1;
                        s.passed += 1;
                    }
                    return Ok(Pass { gate: self.clone() });
                }
                if t0.elapsed() >= self.hold {
                    s.refused += 1;
                    return Err(Refused::Held(t0.elapsed()));
                }
                if !w.counted {
                    // a request that waits is activity: without this, one that left before the
                    // engine came back let the ring be lent again at once
                    w.counted = true;
                    s.waiting += 1;
                    s.last = Instant::now();
                }
            }
            tokio::time::sleep(POLL).await;
        }
    }

    /// Requests wait from now on, if none is in flight or waiting and none came for `idle`.
    pub fn hold(&self, idle: Duration) -> Result<(), String> {
        let mut s = self.st.lock().unwrap();
        if s.held.is_some() {
            return Ok(());
        }
        if s.in_flight > 0 || s.waiting > 0 || s.last.elapsed() < idle {
            return Err(format!("busy: in_flight {} waiting {} idle_s {}", s.in_flight, s.waiting, s.last.elapsed().as_secs()));
        }
        s.held = Some(Instant::now());
        eprintln!("gate: holding requests (the ring is lent; no request for {} s)", s.last.elapsed().as_secs());
        Ok(())
    }

    /// Stop admitting requests and wait for those in flight to finish, so that a steering change lands
    /// at a boundary no request crosses. The guard's drop admits again. A request already in flight is
    /// never cut: it finishes under the alpha it started with.
    pub async fn quiesce(self: &Arc<Gate>, timeout: Duration) -> Result<Quiesced, String> {
        {
            let mut s = self.st.lock().unwrap();
            if s.steering {
                return Err("a steering change is already in progress".into());
            }
            s.steering = true;
        }
        let t0 = Instant::now();
        loop {
            {
                let s = self.st.lock().unwrap();
                if s.in_flight == 0 {
                    return Ok(Quiesced { gate: self.clone() });
                }
            }
            if t0.elapsed() >= timeout {
                let s = self.st.lock().unwrap();
                let msg = format!(
                    "busy: {} request(s) still in flight after {} s",
                    s.in_flight,
                    timeout.as_secs()
                );
                drop(s);
                self.st.lock().unwrap().steering = false;
                return Err(msg);
            }
            tokio::time::sleep(POLL).await;
        }
    }

    pub fn release(&self) {
        let mut s = self.st.lock().unwrap();
        if let Some(t) = s.held.take() {
            eprintln!("gate: released after {} s; {} requests waiting", t.elapsed().as_secs(), s.waiting);
        }
    }

    /// Shutting down: waiting and new requests are answered at once.
    pub fn close(&self) {
        self.st.lock().unwrap().closing = true;
    }

    pub fn status(&self, connected: bool) -> String {
        let s = self.st.lock().unwrap();
        format!(
            "held {} in_flight {} waiting {} idle_s {} engine {} held_s {} passed {} refused {}",
            (s.held.is_some() || s.steering) as u8,
            s.in_flight,
            s.waiting,
            s.last.elapsed().as_secs(),
            connected as u8,
            s.held.map_or(0, |t| t.elapsed().as_secs()),
            s.passed,
            s.refused
        )
    }

    /// Whether a request would pass now, without waiting: the ring not lent, the engine connected.
    pub fn open(&self, connected: bool) -> bool {
        let s = self.st.lock().unwrap();
        !s.closing && s.held.is_none() && !s.steering && connected
    }

    /// Whether requests wait because the ring is lent (for /health).
    pub fn held(&self) -> bool {
        self.st.lock().unwrap().held.is_some()
    }

    /// One admin command: `status`, `hold IDLE_S` or `release`; its one-line answer.
    pub fn command(&self, line: &str, connected: bool) -> String {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w[..] {
            ["status"] => self.status(connected),
            ["hold", n] => match n.parse::<u64>() {
                Ok(n) => self.hold(Duration::from_secs(n)).map_or_else(|e| e, |()| "ok".into()),
                Err(_) => "error: hold IDLE_S (seconds)".into(),
            },
            ["steer"] => "error: steer ALPHA (0.0 to 2.0)".into(),
            ["release"] => {
                self.release();
                "ok".into()
            }
            _ => "error: status | hold IDLE_S | release | steer ALPHA".into(),
        }
    }
}

/// The operator's steering command: `steer ALPHA` sets one global alpha (quiesce, apply, admit again)
/// and `steer` reads the effective state back. The socket is mode 0600 and local: file access is the
/// authorization, exactly as it is for `hold` and `release`. The alpha itself is validated in the
/// engine (the accepted range lives there, model/steer.h); the front-end only refuses to send nonsense.
pub async fn steer(engine: &EngineClient, gate: &Arc<Gate>, line: &str, timeout: Duration) -> String {
    let words: Vec<&str> = line.split_whitespace().collect();
    let set = match words.len() {
        1 => None,
        2 => match words[1].parse::<f32>() {
            Ok(v) if v.is_finite() => Some(v),
            _ => return format!("error: '{}' is not a finite number", words[1]),
        },
        _ => return "error: steer [ALPHA] (0.0 to 2.0)".into(),
    };
    if engine.info().is_none() {
        return "error: the engine is not connected".into();
    }
    let Some(alpha) = set else {
        return match engine.steer_get(timeout).await {
            Ok(st) => steer_line(&st),
            Err(e) => format!("error: {e}"),
        };
    };
    let _guard = match gate.quiesce(timeout).await {
        Ok(g) => g,
        Err(e) => return format!("error: {e}"),
    };
    match engine.steer_set(alpha, timeout).await {
        Ok(st) => steer_line(&st),
        Err(e) => format!("error: {e}"),
    }
}

/// One line: the effective value, its epoch, and every rank's read-back.
fn steer_line(st: &SteerState) -> String {
    let ranks: Vec<String> = st
        .ranks
        .iter()
        .map(|r| {
            if r.ok {
                format!("{}:{}@{}{}", r.rank, r.alpha, r.epoch, if r.why.is_empty() { String::new() } else { format!("({})", r.why) })
            } else {
                format!("{}:refused({})", r.rank, r.why)
            }
        })
        .collect();
    if st.ok {
        format!("ok steering={} alpha={} epoch={} ranks=[{}]", st.on, st.alpha, st.epoch, ranks.join(" "))
    } else {
        format!("error: {} alpha={} epoch={} ranks=[{}]", st.error, st.alpha, st.epoch, ranks.join(" "))
    }
}

/// Serve the admin socket (mode 600; one command per connection, one line back).
pub async fn admin(gate: Arc<Gate>, engine: EngineClient, path: PathBuf) -> std::io::Result<()> {
    // a socket left by an earlier run goes; anything else there is an error
    match std::fs::symlink_metadata(&path) {
        Ok(m) if m.file_type().is_socket() => std::fs::remove_file(&path)?,
        Ok(_) => return Err(std::io::Error::other(format!("{} exists and is not a socket", path.display()))),
        Err(_) => {}
    }
    let l = UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    eprintln!("admin socket {}", path.display());
    loop {
        let (s, _) = l.accept().await?;
        let (gate, engine) = (gate.clone(), engine.clone());
        tokio::spawn(async move {
            let (rd, mut wr) = s.into_split();
            let mut line = String::new();
            let mut rd = BufReader::new(rd.take(256));
            let got = tokio::time::timeout(Duration::from_secs(5), rd.read_line(&mut line)).await;
            let reply = match got {
                Ok(Ok(_)) => {
                    let line = line.trim().to_string();
                    if line == "steer" || line.starts_with("steer ") {
                        // quiesce + engine round-trip: longer than a status line, and bounded
                        steer(&engine, &gate, &line, Duration::from_secs(120)).await
                    } else {
                        gate.command(&line, engine.info().is_some())
                    }
                }
                _ => "error: no command".into(),
            };
            let _ = wr.write_all(format!("{reply}\n").as_bytes()).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn quiesce_drains_and_admits_again() {
        let g = Gate::new(Duration::from_millis(300));
        let on = || true;
        // a request in flight: quiesce waits for it, and a new request waits meanwhile
        let p = g.enter_when(on).await.unwrap();
        let (g2, g3) = (g.clone(), g.clone());
        let q = tokio::spawn(async move { g2.quiesce(Duration::from_secs(5)).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(g3.status(true).starts_with("held 1"), "a steering change holds requests off");
        assert!(!g3.open(true));
        let new = tokio::spawn(async move { g3.enter_when(|| true).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(!q.is_finished(), "quiesce waits for the request in flight");
        drop(p); // the in-flight request finishes under the alpha it started with
        assert_eq!(q.await.unwrap(), Ok(()));
        // the guard dropped: admission is back, and the request that waited passes
        let g4 = g.clone();
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(g4.open(true));
        assert_eq!(new.await.unwrap(), Ok(()));
        assert!(!g.held());
        // a drain that cannot finish in time refuses, and admits again
        let p = g.enter_when(on).await.unwrap();
        assert!(g.quiesce(Duration::from_millis(50)).await.is_err());
        assert!(g.open(true), "a timed-out quiesce does not leave requests held off");
        drop(p);
        // and one at a time
        let p = g.enter_when(on).await.unwrap();
        let first = g.quiesce(Duration::from_secs(5));
        tokio::pin!(first);
        let other = g.quiesce(Duration::from_millis(10)).await;
        assert!(other.is_err(), "a second steering change is refused while one runs");
        drop(p);
        assert!(first.await.is_ok());
    }

    #[tokio::test]
    async fn passes_holds_and_refuses() {
        let g = Gate::new(Duration::from_millis(600));
        let up = Arc::new(AtomicBool::new(true));
        let on = {
            let up = up.clone();
            move || up.load(Ordering::SeqCst)
        };
        // a request in flight: no hold
        let p = g.enter_when(&on).await.unwrap();
        assert!(g.hold(Duration::ZERO).unwrap_err().starts_with("busy: in_flight 1"));
        drop(p);
        // it just ended: not idle for a minute
        assert!(g.hold(Duration::from_secs(60)).is_err());
        assert_eq!(g.command("hold 0", true), "ok");
        assert!(g.status(true).starts_with("held 1 in_flight 0 waiting 0"));
        assert!(g.held());
        // held: a request waits, and passes once released
        let (g2, on2) = (g.clone(), on.clone());
        let t = tokio::spawn(async move { g2.enter_when(&on2).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(g.status(true).contains("waiting 1"));
        assert!(g.hold(Duration::ZERO).is_ok(), "held stays held");
        assert_eq!(g.command("release", true), "ok");
        assert_eq!(t.await.unwrap(), Ok(()));
        assert!(g.status(true).contains("waiting 0") && g.status(true).contains("passed 1"));
        // the engine away: a request waits the hold limit, then is refused
        up.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        assert!(matches!(g.enter_when(&on).await, Err(Refused::Held(_))));
        assert!(t0.elapsed() >= Duration::from_millis(600));
        assert!(g.status(false).contains("refused 1") && g.status(false).contains("waiting 0"));
        // a waiting request whose client goes away is counted out
        let (g3, on3) = (g.clone(), on.clone());
        let t = tokio::spawn(async move { g3.enter_when(&on3).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(g.status(false).contains("waiting 1"));
        t.abort();
        let _ = t.await;
        assert!(g.status(false).contains("waiting 0"));
        // and it was activity: no hold for `idle` after it
        assert!(g.hold(Duration::from_millis(200)).is_err(), "a request that waited and left is recent activity");
        // shutting down: answered at once
        g.close();
        assert_eq!(g.enter_when(&on).await.err(), Some(Refused::Closing));
        assert!(g.command("hold x", true).starts_with("error"));
        assert!(g.command("bogus", true).starts_with("error"));
    }
}
