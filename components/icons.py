"""Line icons (Feather-style paths, 1.75 stroke) as inline SVG. One stroke weight everywhere."""
_PATHS = {
    "mic": '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v4"/>',
    "wave": '<path d="M3 12h2M7 8v8M11 4v16M15 8v8M19 10v4M21 12h0"/>',
    "alert": '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
}


def icon(name: str, size: int = 20) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_PATHS[name]}</svg>')
