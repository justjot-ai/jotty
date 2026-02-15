// Prevents additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{CustomMenuItem, SystemTray, SystemTrayMenu, SystemTrayEvent, Manager};
use tauri::Window;

// Command: Check if running in Tauri
#[tauri::command]
fn is_tauri() -> bool {
    true
}

// Command: Get platform info
#[tauri::command]
fn get_platform() -> String {
    std::env::consts::OS.to_string()
}

// Command: Open external URL
#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open")
        .arg(&url)
        .spawn()
        .map_err(|e| e.to_string())?;

    #[cfg(target_os = "macos")]
    std::process::Command::new("open")
        .arg(&url)
        .spawn()
        .map_err(|e| e.to_string())?;

    #[cfg(target_os = "windows")]
    std::process::Command::new("cmd")
        .args(&["/C", "start", &url])
        .spawn()
        .map_err(|e| e.to_string())?;

    Ok(())
}

// Command: Show notification
#[tauri::command]
fn show_notification(window: Window, title: String, body: String) -> Result<(), String> {
    window
        .emit("notification", serde_json::json!({ "title": title, "body": body }))
        .map_err(|e| e.to_string())?;
    Ok(())
}

// Command: Get app version
#[tauri::command]
fn get_app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Command: Minimize to tray
#[tauri::command]
fn minimize_to_tray(window: Window) -> Result<(), String> {
    window.hide().map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    // System tray menu
    let tray_menu = SystemTrayMenu::new()
        .add_item(CustomMenuItem::new("show".to_string(), "Show Jotty"))
        .add_item(CustomMenuItem::new("new_chat".to_string(), "New Chat"))
        .add_native_item(tauri::SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("quit".to_string(), "Quit"));

    let system_tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .system_tray(system_tray)
        .on_system_tray_event(|app, event| match event {
            SystemTrayEvent::LeftClick {
                position: _,
                size: _,
                ..
            } => {
                let window = app.get_window("main").unwrap();
                window.show().unwrap();
                window.set_focus().unwrap();
            }
            SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
                "show" => {
                    let window = app.get_window("main").unwrap();
                    window.show().unwrap();
                    window.set_focus().unwrap();
                }
                "new_chat" => {
                    let window = app.get_window("main").unwrap();
                    window.show().unwrap();
                    window.set_focus().unwrap();
                    window.emit("new_chat", ()).unwrap();
                }
                "quit" => {
                    std::process::exit(0);
                }
                _ => {}
            },
            _ => {}
        })
        .on_window_event(|event| match event.event() {
            tauri::WindowEvent::CloseRequested { api, .. } => {
                // Hide to tray instead of closing
                event.window().hide().unwrap();
                api.prevent_close();
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            is_tauri,
            get_platform,
            open_url,
            show_notification,
            get_app_version,
            minimize_to_tray,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
