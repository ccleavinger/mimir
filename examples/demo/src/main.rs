#[cfg(feature = "backend")]
use mimir::{launch, mimir_global, GpuArray};
#[cfg(feature = "backend")]
use serde_json::json;
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

type Error = Box<dyn std::error::Error + Send + Sync>;

#[cfg(feature = "backend")]
#[mimir_global]
fn scale(values: &mut [f32], count: i32, factor: f32) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        values[i] *= factor;
    }
}

#[cfg(feature = "backend")]
#[mimir_global]
fn effect(input: &[f32], output: &mut [f32], count: i32, amount: f32) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        output[i] = input[i] * amount;
    }
}

#[derive(Clone)]
struct Target {
    name: String,
    address: String,
}
struct Config {
    targets: Vec<Target>,
}

fn setting(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.into())
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

// This demo accepts bodyless requests only. Read the complete header even when TCP splits it.
fn request(stream: &mut TcpStream) -> Result<(String, String), Error> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    while !bytes.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err("incomplete request".into());
        }
        bytes.extend_from_slice(&chunk[..n]);
        if bytes.len() > 16384 {
            return Err("request header too large".into());
        }
    }
    let text = std::str::from_utf8(&bytes)?;
    let mut parts = text
        .lines()
        .next()
        .ok_or("missing request")?
        .split_whitespace();
    Ok((
        parts.next().ok_or("missing method")?.into(),
        parts.next().ok_or("missing path")?.into(),
    ))
}

#[cfg(feature = "backend")]
fn operation(path: &str) -> Result<Value, Error> {
    match path {
        "/api/status" => {
            let devices = mimir::default_runtime()?.devices()?;
            Ok(
                json!({"devices": devices.iter().map(|d| d.name()).collect::<Vec<_>>(),
                "backend": if cfg!(feature = "cuda") { "CUDA" } else if cfg!(feature = "vulkan") { "Vulkan" } else { "none" },
                "demos": ["vector", "effect"]}),
            )
        }
        "/api/vector" => {
            let values = GpuArray::from_slice(&[1.0f32, 2.0, 3.0, 4.0])?;
            let start = std::time::Instant::now();
            // Four invocations access four distinct elements.
            unsafe {
                launch!(scale<<<[1, 1, 1], [4, 1, 1]>>>(&values, 4, 2.0))?;
            }
            Ok(
                json!({"input": [1,2,3,4], "factor": 2, "output": values.to_vec()?, "elapsed_ms": start.elapsed().as_secs_f64()*1000.0}),
            )
        }
        "/api/effect" => {
            let input = GpuArray::from_slice(&[0.0f32, 0.25, 0.5, 0.75, 1.0])?;
            let output = GpuArray::from_slice(&[0.0f32; 5])?;
            // Five invocations read and write distinct in-bounds elements.
            unsafe {
                launch!(effect<<<[1, 1, 1], [5, 1, 1]>>>(&input, &output, 5, 0.8))?;
            }
            Ok(json!({"input": [0.0,0.25,0.5,0.75,1.0], "amount": 0.8, "output": output.to_vec()?}))
        }
        _ => Err("unknown operation".into()),
    }
}

fn remote(target: &Target, method: &str, operation: &str) -> Result<Value, Error> {
    let addr = target
        .address
        .to_socket_addrs()?
        .next()
        .ok_or("backend address not found")?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    write!(stream, "{method} /api/{operation} HTTP/1.1\r\nHost: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", target.address)?;
    let mut bytes = Vec::new();
    stream.take(1024 * 1024).read_to_end(&mut bytes)?;
    let text = std::str::from_utf8(&bytes)?;
    let (headers, body) = text
        .split_once("\r\n\r\n")
        .ok_or("invalid backend response")?;
    let value: Value = serde_json::from_str(body)?;
    if !headers.starts_with("HTTP/1.1 200 ") {
        return Err(value["error"].as_str().unwrap_or("backend failed").into());
    }
    Ok(value)
}

fn serve(mut stream: TcpStream, ui: bool, config: &Config) -> Result<(), Error> {
    let (method, path) = request(&mut stream)?;
    let path = path.split('?').next().unwrap_or(&path);
    if ui && method == "GET" && path == "/" {
        let options = config
            .targets
            .iter()
            .enumerate()
            .map(|(i, t)| format!("<option value=\"{i}\">{}</option>", escape(&t.name)))
            .collect::<String>();
        return Ok(response(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            &include_str!("page.html").replace("<!-- targets -->", &options),
        )?);
    }
    if ui && path.starts_with("/run/") {
        let parts: Vec<_> = path.trim_start_matches("/run/").split('/').collect();
        if parts.len() == 2 && ["status", "vector", "effect"].contains(&parts[1]) {
            let expected = if parts[1] == "status" { "GET" } else { "POST" };
            if method != expected {
                return Ok(response(
                    &mut stream,
                    "405 Method Not Allowed",
                    "text/plain",
                    "method not allowed",
                )?);
            }
            let result = parts[0]
                .parse::<usize>()
                .ok()
                .and_then(|i| config.targets.get(i))
                .ok_or_else(|| "unknown target".into())
                .and_then(|t| remote(t, &method, parts[1]));
            let body = match result {
                Ok(value) => format!(
                    "<pre>{}</pre>",
                    escape(&serde_json::to_string_pretty(&value)?)
                ),
                Err(e) => format!(
                    "<p role=\"alert\">Backend unavailable: {}</p>",
                    escape(&e.to_string())
                ),
            };
            return Ok(response(
                &mut stream,
                "200 OK",
                "text/html; charset=utf-8",
                &body,
            )?);
        }
    }
    #[cfg(feature = "backend")]
    if !ui && ["/api/status", "/api/vector", "/api/effect"].contains(&path) {
        let expected = if path == "/api/status" { "GET" } else { "POST" };
        if method != expected {
            return Ok(response(
                &mut stream,
                "405 Method Not Allowed",
                "application/json",
                "{\"error\":\"method not allowed\"}",
            )?);
        }
        let (status, value) = match operation(path) {
            Ok(value) => ("200 OK", value),
            Err(e) => ("503 Service Unavailable", json!({"error": e.to_string()})),
        };
        return Ok(response(
            &mut stream,
            status,
            "application/json",
            &value.to_string(),
        )?);
    }
    Ok(response(
        &mut stream,
        "404 Not Found",
        "text/plain",
        "not found",
    )?)
}

fn listen(listener: TcpListener, ui: bool, config: Arc<Config>) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let config = config.clone();
                std::thread::spawn(move || {
                    if let Err(e) = serve(stream, ui, &config) {
                        eprintln!("request failed: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept failed: {e}"),
        }
    }
}

fn main() -> Result<(), Error> {
    if !cfg!(any(feature = "ui", feature = "backend")) {
        return Err("enable ui or backend".into());
    }
    // Environment overrides optional compile-time defaults, so one binary can be reused.
    let api_addr = setting(
        "MIMIR_DEMO_API_ADDR",
        option_env!("MIMIR_DEMO_API_ADDR").unwrap_or("127.0.0.1:8081"),
    );
    let ui_addr = setting(
        "MIMIR_DEMO_UI_ADDR",
        option_env!("MIMIR_DEMO_UI_ADDR").unwrap_or("127.0.0.1:8080"),
    );
    let defaults = if cfg!(feature = "backend") {
        format!("Laptop=http://{api_addr}")
    } else {
        String::new()
    };
    let targets = setting(
        "MIMIR_DEMO_TARGETS",
        option_env!("MIMIR_DEMO_TARGETS").unwrap_or(&defaults),
    );
    let targets = targets
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let (name, url) = s
                .trim()
                .split_once('=')
                .ok_or("targets must be Name=http://host:port")?;
            let address = url
                .strip_prefix("http://")
                .ok_or("targets support http:// only")?
                .trim_end_matches('/');
            if name.is_empty() || address.is_empty() || address.contains('/') {
                return Err("invalid target".into());
            }
            Ok(Target {
                name: name.into(),
                address: address.into(),
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let config = Arc::new(Config { targets });
    // Bind both before starting either, so address conflicts fail at startup.
    let api = if cfg!(feature = "backend") {
        Some(TcpListener::bind(&api_addr)?)
    } else {
        None
    };
    let ui = if cfg!(feature = "ui") {
        Some(TcpListener::bind(&ui_addr)?)
    } else {
        None
    };
    let mut threads = Vec::new();
    if let Some(listener) = api {
        println!("GPU API: http://{api_addr}");
        let config = config.clone();
        threads.push(std::thread::spawn(move || listen(listener, false, config)));
    }
    if let Some(listener) = ui {
        println!("Demo UI: http://{ui_addr}");
        threads.push(std::thread::spawn(move || listen(listener, true, config)));
    }
    for thread in threads {
        thread.join().map_err(|_| "server thread panicked")?;
    }
    Ok(())
}
