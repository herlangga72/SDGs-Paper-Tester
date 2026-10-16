use std::fmt::Display;
use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU8, Ordering};

const BACKEND_IO_URING: u8 = 1;
static ACTIVE_BACKEND: AtomicU8 = AtomicU8::new(0);

pub(super) fn active_backend_name() -> &'static str {
    match ACTIVE_BACKEND.load(Ordering::Relaxed) {
        BACKEND_IO_URING => "io_uring",
        _ => "threaded",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BackendKind {
    IoUring,
    Threaded,
}

impl BackendKind {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::IoUring => "io_uring",
            Self::Threaded => "threaded",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum BackendSelection<T> {
    IoUring(T),
    Threaded { reason: Option<String> },
}

impl<T> BackendSelection<T> {
    pub(super) const fn kind(&self) -> BackendKind {
        match self {
            Self::IoUring(_) => BackendKind::IoUring,
            Self::Threaded { .. } => BackendKind::Threaded,
        }
    }

    pub(super) fn fallback_reason(&self) -> Option<&str> {
        match self {
            Self::IoUring(_) => None,
            Self::Threaded { reason } => reason.as_deref(),
        }
    }
}

pub(super) fn select_backend<T, E>(
    enabled: bool,
    probe: impl FnOnce() -> Result<T, E>,
) -> BackendSelection<T>
where
    E: Display,
{
    if !enabled {
        return BackendSelection::Threaded { reason: None };
    }

    match probe() {
        Ok(backend) => BackendSelection::IoUring(backend),
        Err(error) => BackendSelection::Threaded {
            reason: Some(error.to_string()),
        },
    }
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub(super) fn detect_backend() -> BackendSelection<io_uring::IoUring> {
    select_backend(true, create_ring)
}

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
pub(super) fn detect_backend() -> BackendSelection<()> {
    select_backend(false, || Ok::<(), io::Error>(()))
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
fn create_ring() -> io::Result<io_uring::IoUring> {
    use io_uring::{opcode, IoUring, Probe};

    let ring = IoUring::new(256)?;
    let mut probe = Probe::new();
    ring.submitter().register_probe(&mut probe)?;
    if !probe.is_supported(opcode::Accept::CODE) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "kernel does not support the io_uring accept opcode",
        ));
    }
    Ok(ring)
}

fn spawn_connection(stream: TcpStream, handler: fn(&mut TcpStream)) {
    std::thread::spawn(move || {
        let mut stream = stream;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handler(&mut stream);
        }));
    });
}

pub(super) fn run_threaded(listener: TcpListener, handler: fn(&mut TcpStream)) -> ! {
    ACTIVE_BACKEND.store(0, Ordering::Relaxed);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => spawn_connection(stream, handler),
            Err(error) => eprintln!("[web] accept error: {error}"),
        }
    }
    unreachable!("TcpListener::incoming() does not terminate while the listener is owned")
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
fn accept_one(listener: &TcpListener, ring: &mut io_uring::IoUring) -> io::Result<TcpStream> {
    use io_uring::{opcode, types};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::ptr;

    const ACCEPT_USER_DATA: u64 = 0x5344_4741;
    let entry = opcode::Accept::new(
        types::Fd(listener.as_raw_fd()),
        ptr::null_mut(),
        ptr::null_mut(),
    )
    .build()
    .user_data(ACCEPT_USER_DATA);

    {
        let mut submission = ring.submission();
        // SAFETY: the accept SQE contains only the listener fd and null address pointers. The
        // listener remains alive for the full ring loop, and the entry is copied into the SQ.
        unsafe {
            submission.push(&entry).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "io_uring submission queue is full",
                )
            })?;
        }
    }

    ring.submit_and_wait(1)?;
    let completion = ring
        .completion()
        .next()
        .ok_or_else(|| io::Error::other("io_uring accept completed without a CQE"))?;
    if completion.user_data() != ACCEPT_USER_DATA {
        return Err(io::Error::other(
            "unexpected io_uring completion for accept loop",
        ));
    }
    let accepted_fd = completion.result();
    if accepted_fd < 0 {
        return Err(io::Error::from_raw_os_error(accepted_fd.saturating_neg()));
    }

    // SAFETY: a successful Accept CQE returns a newly-owned socket fd. Ownership transfers once
    // into TcpStream, which closes it when dropped.
    Ok(unsafe { TcpStream::from_raw_fd(accepted_fd) })
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub(super) fn run_selected(
    listener: TcpListener,
    selection: BackendSelection<io_uring::IoUring>,
    handler: fn(&mut TcpStream),
) -> ! {
    match selection {
        BackendSelection::IoUring(mut ring) => {
            ACTIVE_BACKEND.store(BACKEND_IO_URING, Ordering::Relaxed);
            eprintln!("[web] io_uring accept backend active");
            loop {
                match accept_one(&listener, &mut ring) {
                    Ok(stream) => spawn_connection(stream, handler),
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) if error.kind() == io::ErrorKind::ConnectionAborted => continue,
                    Err(error) => {
                        eprintln!("[web] io_uring accept failed, falling back to threaded backend: {error}");
                        drop(ring);
                        run_threaded(listener, handler);
                    }
                }
            }
        }
        BackendSelection::Threaded { reason } => {
            if let Some(reason) = reason {
                eprintln!("[web] io_uring unavailable, using threaded backend: {reason}");
            }
            run_threaded(listener, handler)
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
pub(super) fn run_selected(
    listener: TcpListener,
    selection: BackendSelection<()>,
    handler: fn(&mut TcpStream),
) -> ! {
    ACTIVE_BACKEND.store(0, Ordering::Relaxed);
    if let BackendSelection::Threaded {
        reason: Some(reason),
    } = selection
    {
        eprintln!("[web] io_uring unavailable, using threaded backend: {reason}");
    }
    run_threaded(listener, handler)
}

#[cfg(test)]
mod tests {
    use super::{active_backend_name, select_backend, BackendKind, BackendSelection};
    use std::cell::Cell;

    #[test]
    fn disabled_feature_selects_threaded_without_probing() {
        let probed = Cell::new(false);
        let selected = select_backend(false, || {
            probed.set(true);
            Ok::<usize, &'static str>(7)
        });
        assert!(matches!(
            selected,
            BackendSelection::Threaded { reason: None }
        ));
        assert!(!probed.get());
        assert_eq!(active_backend_name(), "threaded");
    }

    #[test]
    fn successful_probe_selects_io_uring_backend() {
        assert_eq!(BackendKind::IoUring.name(), "io_uring");
        assert_eq!(BackendKind::Threaded.name(), "threaded");
        let selected = select_backend(true, || Ok::<usize, &'static str>(7));
        match selected {
            BackendSelection::IoUring(value) => assert_eq!(value, 7),
            BackendSelection::Threaded { .. } => panic!("successful probe must select io_uring"),
        }
    }

    #[test]
    fn failed_probe_selects_threaded_with_reason() {
        let selected = select_backend(true, || Err::<usize, _>("operation not permitted"));
        match selected {
            BackendSelection::Threaded { reason } => {
                assert_eq!(reason.as_deref(), Some("operation not permitted"));
            }
            BackendSelection::IoUring(_) => panic!("failed probe must select threaded backend"),
        }
    }
}
