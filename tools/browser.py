"""Browser automation tool using Playwright."""

import json
import asyncio
from pathlib import Path

# Global browser state (persistent across tool calls)
_browser = None
_page = None


async def _get_browser():
    """Get or create a persistent browser instance."""
    global _browser, _page
    if _browser is None:
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        _browser = await pw.chromium.launch(headless=True)
        _page = await _browser.new_page()
    return _browser, _page


def browser_navigate(url: str) -> str:
    """Navigate the browser to a URL."""
    async def _nav():
        _, page = await _get_browser()
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        title = await page.title()
        return f"Navigated to: {url}\nTitle: {title}"
    return asyncio.run(_nav())


def browser_snapshot(max_chars: int = 5000) -> str:
    """Get a text snapshot of the current page (readable content)."""
    async def _snap():
        _, page = await _get_browser()
        text = await page.inner_text("body")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text
    return asyncio.run(_snap())


def browser_click(selector: str) -> str:
    """Click an element by CSS selector."""
    async def _click():
        _, page = await _get_browser()
        await page.click(selector, timeout=5000)
        return f"Clicked: {selector}"
    return asyncio.run(_click())


def browser_type(selector: str, text: str, press_enter: bool = False) -> str:
    """Type text into an element. Optionally press Enter after."""
    async def _type():
        _, page = await _get_browser()
        await page.fill(selector, "", timeout=5000)
        await page.type(selector, text, delay=50)
        if press_enter:
            await page.press(selector, "Enter")
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        return f"Typed '{text}' into {selector}"
    return asyncio.run(_type())


def browser_screenshot(path: str = "/tmp/browser_screenshot.png") -> str:
    """Take a screenshot of the current page."""
    async def _screenshot():
        _, page = await _get_browser()
        await page.screenshot(path=path)
        return f"Screenshot saved to: {path}"
    return asyncio.run(_screenshot())


def browser_list_elements() -> str:
    """List clickable/interactive elements on the page."""
    async def _list():
        _, page = await _get_browser()
        elements = await page.evaluate("""() => {
            const items = [];
            document.querySelectorAll('a, button, input, textarea, select, [role="button"]').forEach((el, i) => {
                if (i >= 30) return;
                const tag = el.tagName.toLowerCase();
                const text = el.textContent?.trim().substring(0, 60) || '';
                const href = el.href || '';
                const type = el.type || '';
                const placeholder = el.placeholder || '';
                items.push({tag, text, href, type, placeholder, selector: el.tagName + (el.id ? '#' + el.id : '')});
            });
            return items;
        }""")
        return json.dumps(elements, indent=2)
    return asyncio.run(_list())


def browser_close() -> str:
    """Close the browser."""
    async def _close():
        global _browser, _page
        if _browser:
            await _browser.close()
            _browser = None
            _page = None
        return "Browser closed"
    return asyncio.run(_close())
