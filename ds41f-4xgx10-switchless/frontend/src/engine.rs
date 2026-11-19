//! Client side of the rank-0 socket: one connection, many requests multiplexed by req_id.
//! Reconnects with backoff; requests in flight when the connection drops end with an error.

use crate::protocol::{read_frame, Done, DoneReason, Frame, HelloEngine, SteerState, Submit, Tokens, VERSION};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot, watch};

#[derive(Debug)]
pub enum Event {
    Tokens(Tokens),
    Done(Done),
}

struct Pending {
    tx: mpsc::UnboundedSender<Event>,
    n_top: u32,
}

#[derive(Default)]
struct Conn {
    tx: Option<mpsc::UnboundedSender<Vec<u8>>>,
    pending: HashMap<u64, Pending>,
    /// steering requests in flight, by req_id
    steering: HashMap<u64, SteerTx>,
}

struct Inner {
    path: PathBuf,
    conn: Mutex<Conn>,
    next_req: AtomicU64,
    next_steer: AtomicU64,
    info: watch::Sender<Option<HelloEngine>>,
}

#[derive(Clone)]
pub struct EngineClient {
    inner: Arc<Inner>,
}

/// The steering control's answer to one request (STEER_STATE arrives on the same connection).
type SteerTx = oneshot::Sender<SteerState>;

impl EngineClient {
    /// Start connecting to the engine socket (in the background, forever).
    pub fn start(path: PathBuf) -> EngineClient {
        let (info, _) = watch::channel(None);
        let inner = Arc::new(Inner {
            path,
            conn: Mutex::new(Conn::default()),
            next_req: AtomicU64::new(1),
            next_steer: AtomicU64::new(1),
            info,
        });
        tokio::spawn(run(inner.clone()));
        EngineClient { inner }
    }

    /// The engine's HELLO while connected.
    pub fn info(&self) -> Option<HelloEngine> {
        self.inner.info.borrow().clone()
    }

    pub async fn wait_connected(&self, timeout: Duration) -> Option<HelloEngine> {
        let mut rx = self.inner.info.subscribe();
        let _ = tokio::time::timeout(timeout, rx.wait_for(|i| i.is_some())).await;
        self.info()
    }

    /// Send a request (its req_id is assigned here). Events arrive on the receiver; the last one is
    /// always Done.
    pub fn submit(&self, mut s: Submit) -> Result<(u64, mpsc::UnboundedReceiver<Event>), String> {
        s.req_id = self.inner.next_req.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        let mut c = self.inner.conn.lock().unwrap();
        let wtx = c.tx.clone().ok_or("engine not connected")?;
        c.pending.insert(s.req_id, Pending { tx, n_top: if s.flags & crate::protocol::FLAG_LOGPROBS != 0 { s.n_top_logprobs } else { 0 } });
        if wtx.send(Frame::Submit(s.clone()).encode()).is_err() {
            c.pending.remove(&s.req_id);
            return Err("engine not connected".into());
        }
        Ok((s.req_id, rx))
    }

    /// Set one global alpha for all traffic (docs/steering-control.md): the engine applies it at a step
    /// boundary on every rank and answers with each rank's read-back. The caller is responsible for
    /// having stopped admitting requests first (gate::quiesce).
    pub async fn steer_set(&self, alpha: f32, timeout: Duration) -> Result<SteerState, String> {
        self.steer(Frame::SteerSet { req_id: 0, alpha }, timeout).await
    }

    /// Read every device; the server may complete pending recovery before returning a safe result.
    pub async fn steer_get(&self, timeout: Duration) -> Result<SteerState, String> {
        self.steer(Frame::SteerGet { req_id: 0 }, timeout).await
    }

    async fn steer(&self, frame: Frame, timeout: Duration) -> Result<SteerState, String> {
        let id = self.inner.next_steer.fetch_add(1, Ordering::Relaxed);
        let frame = match frame {
            Frame::SteerSet { alpha, .. } => Frame::SteerSet { req_id: id, alpha },
            Frame::SteerGet { .. } => Frame::SteerGet { req_id: id },
            _ => return Err("not a steering frame".into()),
        };
        let (tx, rx) = oneshot::channel();
        {
            let mut c = self.inner.conn.lock().unwrap();
            let wtx = c.tx.clone().ok_or("engine not connected")?;
            c.steering.insert(id, tx);
            if wtx.send(frame.encode()).is_err() {
                c.steering.remove(&id);
                return Err("engine not connected".into());
            }
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(st)) => Ok(st),
            Ok(Err(_)) => Err("the engine connection dropped the steering request".into()),
            Err(_) => {
                self.inner.conn.lock().unwrap().steering.remove(&id);
                Err(format!("no answer from the engine in {} s", timeout.as_secs()))
            }
        }
    }

    pub fn cancel(&self, req_id: u64) {
        self.send(Frame::Cancel { req_id });
    }

    pub fn release(&self, session_id: u64) {
        self.send(Frame::Release { session_id });
    }

    fn send(&self, f: Frame) {
        if let Some(tx) = &self.inner.conn.lock().unwrap().tx {
            let _ = tx.send(f.encode());
        }
    }
}

async fn run(inner: Arc<Inner>) {
    let mut backoff = Duration::from_millis(100);
    let mut logged_down = false;
    loop {
        match UnixStream::connect(&inner.path).await {
            Ok(stream) => {
                backoff = Duration::from_millis(100);
                logged_down = false;
                match connection(&inner, stream).await {
                    Ok(()) => eprintln!("engine: connection closed"),
                    Err(e) => eprintln!("engine: connection lost: {e}"),
                }
                inner.info.send_replace(None);
                let (pending, steering) = {
                    let mut c = inner.conn.lock().unwrap();
                    c.tx = None;
                    (std::mem::take(&mut c.pending), std::mem::take(&mut c.steering))
                };
                drop(steering);  // their receivers see the cancel
                for (req_id, p) in pending {
                    let _ = p.tx.send(Event::Done(Done {
                        req_id,
                        reason: DoneReason::Error,
                        prompt_tokens: 0,
                        cached_tokens: 0,
                        completion_tokens: 0,
                        queue_us: 0,
                        prefill_us: 0,
                        decode_us: 0,
                        error: "engine connection lost".into(),
                    }));
                }
            }
            Err(e) => {
                if !logged_down {
                    eprintln!("engine: cannot connect to {}: {e}; retrying", inner.path.display());
                    logged_down = true;
                }
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(2));
    }
}

async fn connection(inner: &Arc<Inner>, stream: UnixStream) -> Result<(), String> {
    let (mut rd, mut wr) = stream.into_split();
    wr.write_all(&Frame::Hello { version: VERSION }.encode()).await.map_err(|e| e.to_string())?;
    let (ty, p) = read_frame(&mut rd).await.map_err(|e| e.to_string())?.ok_or("closed before HELLO")?;
    let hello = match Frame::decode(ty, &p, |_| 0)? {
        Frame::HelloEngine(h) => h,
        f => return Err(format!("expected HELLO, got {f:?}")),
    };
    if hello.version != VERSION {
        return Err(format!("engine speaks protocol version {}, we speak {VERSION}", hello.version));
    }
    eprintln!(
        "engine: connected to {} ({}; context {}, vocab {}, eos {}, {} sessions)",
        inner.path.display(),
        hello.name,
        hello.max_context,
        hello.vocab_size,
        hello.eos_id,
        hello.max_sessions
    );
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    inner.conn.lock().unwrap().tx = Some(tx);
    inner.info.send_replace(Some(hello));
    let writer = tokio::spawn(async move {
        while let Some(b) = rx.recv().await {
            if wr.write_all(&b).await.is_err() {
                break;
            }
        }
    });
    let result = loop {
        let (ty, p) = match read_frame(&mut rd).await {
            Ok(Some(f)) => f,
            Ok(None) => break Ok(()),
            Err(e) => break Err(e.to_string()),
        };
        let frame = {
            let c = inner.conn.lock().unwrap();
            Frame::decode(ty, &p, |id| c.pending.get(&id).map_or(0, |p| p.n_top))
        };
        match frame {
            Ok(Frame::Tokens(t)) => {
                let c = inner.conn.lock().unwrap();
                if let Some(p) = c.pending.get(&t.req_id) {
                    let _ = p.tx.send(Event::Tokens(t));
                }
            }
            Ok(Frame::SteerState(st)) => {
                let tx = inner.conn.lock().unwrap().steering.remove(&st.req_id);
                if let Some(tx) = tx {
                    let _ = tx.send(st);
                }
            }
            Ok(Frame::Done(d)) => {
                let p = inner.conn.lock().unwrap().pending.remove(&d.req_id);
                if let Some(p) = p {
                    let _ = p.tx.send(Event::Done(d));
                }
            }
            Ok(f) => break Err(format!("unexpected frame from engine: {f:?}")),
            Err(e) => break Err(e),
        }
    };
    writer.abort();
    result
}
