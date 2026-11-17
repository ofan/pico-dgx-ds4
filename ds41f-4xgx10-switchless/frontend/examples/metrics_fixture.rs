//! CPU telemetry-isolation fixture, NOT a GPU serving qualification.
use dsfront::metrics::{Endpoint, M};
use std::sync::atomic::Ordering::Relaxed;
fn main() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let _endpoint = Endpoint::start(addr, "telemetry-test".into()).unwrap();
    println!("{}", addr.port());
    std::thread::sleep(std::time::Duration::from_secs(3));
    M.running.store(16, Relaxed);
    M.waiting.store(3, Relaxed);
    M.generation_tokens.store(32, Relaxed);
    std::thread::sleep(std::time::Duration::from_secs(125));
    M.running.store(0, Relaxed);
    M.waiting.store(0, Relaxed);
    M.engine_done[4].fetch_add(1, Relaxed);
    std::thread::sleep(std::time::Duration::from_secs(3));
}
