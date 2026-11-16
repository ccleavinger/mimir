#[cfg(feature = "backend")]
mod gpu;
#[cfg(feature = "backend")]
mod mnist;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;
type Error = Box<dyn std::error::Error + Send + Sync>;

// A BMP data URL keeps image transport self-contained without an image codec dependency.
#[cfg(feature = "backend")]
fn bitmap(width: u32, height: u32, pixels: &[f32]) -> String {
    let size = 54 + width * height * 4;
    let mut bytes = Vec::with_capacity(size as usize);
    bytes.extend_from_slice(b"BM");
    for n in [size, 0, 54, 40, width, height] {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    bytes.extend_from_slice(&[1, 0, 32, 0]);
    for n in [0u32, width * height * 4, 2835, 2835, 0, 0] {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    for y in (0..height as usize).rev() {
        for x in 0..width as usize {
            let i = (y * width as usize + x) * 4;
            bytes.extend_from_slice(&[
                pixels[i + 2] as u8,
                pixels[i + 1] as u8,
                pixels[i] as u8,
                255,
            ]);
        }
    }
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::from("data:image/bmp;base64,");
    for c in bytes.chunks(3) {
        let n = ((c[0] as u32) << 16)
            | ((c.get(1).copied().unwrap_or(0) as u32) << 8)
            | c.get(2).copied().unwrap_or(0) as u32;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if c.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
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

// Content-Length framing is sufficient for browser form and demo API requests.
fn request(stream: &mut TcpStream) -> Result<(String, String, Vec<u8>), Error> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut bytes = Vec::new();
    let mut chunk = [0; 8192];
    let end = loop {
        if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err("incomplete request".into());
        }
        bytes.extend_from_slice(&chunk[..n]);
        if bytes.len() > 32768 {
            return Err("header too large".into());
        }
    };
    let header = std::str::from_utf8(&bytes[..end])?;
    let mut parts = header
        .lines()
        .next()
        .ok_or("missing request")?
        .split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_owned();
    let path = parts.next().ok_or("missing path")?.to_owned();
    let length = header
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .map(|(_, v)| v.trim().parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if length > 16 * 1024 * 1024 {
        return Err("image request too large".into());
    }
    while bytes.len() < end + length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err("incomplete body".into());
        }
        bytes.extend_from_slice(&chunk[..n]);
    }
    Ok((method, path, bytes[end..end + length].to_vec()))
}

fn form_args(body: &[u8]) -> Result<Value, Error> {
    let mut args = json!({});
    for pair in std::str::from_utf8(body)?
        .split('&')
        .filter(|s| !s.is_empty())
    {
        let (k, v) = pair.split_once('=').ok_or("invalid form")?;
        let mut decoded = Vec::new();
        let mut i = 0;
        let b = v.as_bytes();
        while i < b.len() {
            if b[i] == b'%' {
                if i + 2 >= b.len() {
                    return Err("invalid escape".into());
                }
                decoded.push(u8::from_str_radix(
                    std::str::from_utf8(&b[i + 1..i + 3])?,
                    16,
                )?);
                i += 3;
            } else {
                decoded.push(if b[i] == b'+' { b' ' } else { b[i] });
                i += 1;
            }
        }
        args[k] = serde_json::from_slice(&decoded)?;
    }
    Ok(args)
}

fn remote(target: &Target, method: &str, operation: &str, args: &Value) -> Result<Value, Error> {
    let addr = target
        .address
        .to_socket_addrs()?
        .next()
        .ok_or("backend address not found")?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let payload = args.to_string();
    write!(stream, "{method} /api/{operation} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", target.address, payload.len(), payload)?;
    let mut bytes = Vec::new();
    stream.take(16 * 1024 * 1024).read_to_end(&mut bytes)?;
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
    let (method, path, payload) = match request(&mut stream) {
        Ok(request) => request,
        Err(e) => {
            return Ok(response(
                &mut stream,
                "400 Bad Request",
                "application/json",
                &json!({"error": e.to_string()}).to_string(),
            )?)
        }
    };
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
        if parts.len() == 2 && ["status", "fractal", "blur", "mnist"].contains(&parts[1]) {
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
                .and_then(|t| remote(t, &method, parts[1], &form_args(&payload)?));
            let body = match result {
                Ok(value) => {
                    if let Some(prediction) = value["prediction"].as_u64() {
                        let scores = value["probabilities"]
                            .as_array()
                            .ok_or("missing probabilities")?;
                        let bars = scores.iter().enumerate().map(|(digit, score)| {
                            let pct = score.as_f64().unwrap_or(0.0)*100.0;
                            format!("<div class=\"score\"><span>{digit}</span><meter min=\"0\" max=\"100\" value=\"{pct:.2}\">{pct:.1}%</meter><span>{pct:.1}%</span></div>")
                        }).collect::<String>();
                        format!("<div class=\"prediction\"><div><span class=\"eyebrow\">Prediction</span><div class=\"digit\">{prediction}</div><p>{:.1} ms on backend</p><img class=\"digit-preview\" src=\"{}\" alt=\"28 by 28 input seen by the model\"><p>What the model sees</p></div><div class=\"scores\">{bars}</div></div>",value["elapsed_ms"].as_f64().unwrap_or(0.0),escape(value["image"].as_str().unwrap_or("")))
                    } else if let Some(image) = value["image"].as_str() {
                        format!("<figure><img src=\"{}\" alt=\"GPU rendered result\"><figcaption>{} · {} × {} · {:.1} ms on backend (includes allocation, first-use compilation and readback)</figcaption></figure>",escape(image),escape(value["detail"].as_str().unwrap_or("")),value["width"],value["height"],value["elapsed_ms"].as_f64().unwrap_or(0.0))
                    } else {
                        format!(
                            "<span class=\"device\">{} · {}</span>",
                            escape(value["backend"].as_str().unwrap_or("GPU")),
                            escape(
                                &value["devices"]
                                    .as_array()
                                    .map(|d| d
                                        .iter()
                                        .filter_map(|v| v.as_str())
                                        .collect::<Vec<_>>()
                                        .join(", "))
                                    .unwrap_or_else(|| "No device".into())
                            )
                        )
                    }
                }
                Err(e) => format!(
                    "<p role=\"alert\">Request failed: {}</p>",
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
    if !ui && ["/api/status", "/api/fractal", "/api/blur", "/api/mnist"].contains(&path) {
        let expected = if path == "/api/status" { "GET" } else { "POST" };
        if method != expected {
            return Ok(response(
                &mut stream,
                "405 Method Not Allowed",
                "application/json",
                "{\"error\":\"method not allowed\"}",
            )?);
        }
        let (status, value) = match serde_json::from_slice::<Value>(&payload)
            .or_else(|e| {
                if payload.is_empty() {
                    Ok(json!({}))
                } else {
                    Err(e)
                }
            })
            .map_err(Error::from)
            .and_then(|args| gpu::run(path, &args))
        {
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
    #[cfg(feature = "backend")]
    mnist::init()?;
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
