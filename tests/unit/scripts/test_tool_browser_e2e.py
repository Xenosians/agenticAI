"""Offline acceptance observations with disposable browser/server fixtures."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import tool_browser_e2e as browser


class FakePage:
    def __init__(self):
        self.elapsed = 0.0
        self.waits = []
        self.after_wait = lambda: None

    def on(self, event, callback):
        pass

    def wait_for_timeout(self, milliseconds):
        self.waits.append(milliseconds)
        self.elapsed += milliseconds / 1000
        self.after_wait()


@pytest.fixture
def observations(monkeypatch):
    page = FakePage()
    monkeypatch.setattr(browser.time, "monotonic", lambda: page.elapsed)
    traffic = browser.BrowserTraffic(page)
    return page, traffic


def job(status, approval_id="source-approval"):
    return {"job_id": "fixture-job", "status": status,
            "proposed_tool": {"approval_id": approval_id}}


@pytest.mark.parametrize("status", sorted(browser.TERMINAL))
def test_requested_terminal_returns_immediately_with_retained_source_approval(observations, status):
    page, traffic = observations
    terminal = job(status)
    traffic.jobs["fixture-job"] = terminal

    actual = traffic.wait(page, "http://fixture.invalid", "fixture-job",
                          browser.TERMINAL | {"waiting_approval"}, 2,
                          approval_other_than="source-approval")

    assert actual is terminal
    assert page.waits == []


def test_same_waiting_approval_is_ignored_until_distinct_next_approval(observations):
    page, traffic = observations
    traffic.jobs["fixture-job"] = job("waiting_approval")
    next_step = job("waiting_approval", "ticket-approval")
    page.after_wait = lambda: traffic.jobs.update({"fixture-job": next_step})

    actual = traffic.wait(page, "http://fixture.invalid", "fixture-job",
                          browser.TERMINAL | {"waiting_approval"}, 2,
                          approval_other_than="source-approval")

    assert actual is next_step
    assert page.waits == [500]


def test_same_waiting_approval_exhausts_declared_budget_without_being_returned(observations):
    page, traffic = observations
    traffic.jobs["fixture-job"] = job("waiting_approval")

    with pytest.raises(TimeoutError, match="last=waiting_approval"):
        traffic.wait(page, "http://fixture.invalid", "fixture-job",
                     browser.TERMINAL | {"waiting_approval"}, 1,
                     approval_other_than="source-approval")

    assert page.waits == [500, 500]


def test_unrequested_terminal_state_does_not_expand_the_requested_scope(observations):
    page, traffic = observations
    traffic.jobs["fixture-job"] = job("failed")

    with pytest.raises(TimeoutError, match="last=failed"):
        traffic.wait(page, "http://fixture.invalid", "fixture-job", {"waiting_approval"}, 1,
                     approval_other_than="source-approval")

    assert page.waits == [500, 500]


def test_refresh_observes_terminal_failure_without_waiting_for_another_approval(observations, monkeypatch):
    page, traffic = observations
    traffic.jobs["fixture-job"] = job("processing")
    failed = job("failed")
    refreshes = []

    def refresh(observed_page, backend, job_id):
        assert observed_page is page
        assert backend == "http://fixture.invalid"
        refreshes.append(job_id)
        traffic.jobs[job_id] = failed

    monkeypatch.setattr(traffic, "refresh", refresh)
    actual = traffic.wait(page, "http://fixture.invalid", "fixture-job",
                          browser.TERMINAL | {"waiting_approval"}, 5,
                          approval_other_than="source-approval")

    assert actual is failed
    assert refreshes == ["fixture-job"]
    assert page.waits == [500] * 7


def test_initial_wait_without_approval_qualifier_keeps_existing_behavior(observations):
    page, traffic = observations
    waiting = job("waiting_approval")
    traffic.jobs["fixture-job"] = waiting

    assert traffic.wait(page, "http://fixture.invalid", "fixture-job", {"waiting_approval"}, 1) is waiting
    assert page.waits == []


# The live workspace has two buttons named exactly "New chat" once the header
# menu is open: the menu's command and any saved chat that kept its default
# title (agenticFrontend src/agenticFrontend.nim renderHeaderMenu,
# src/components/sidebar.nim renderSavedChat; its icon is aria-hidden).
WORKSPACE_WITH_A_SAVED_CHAT_NAMED_NEW_CHAT = """
<aside class="sidebar">
  <button class="new-session"><span>+</span>New chat</button>
  <nav class="chat-nav-section">
    <div class="chat-nav-row">
      <button class="chat-nav-main" onclick="window.clicks.push('saved chat')">
        <span class="nav-icon" aria-hidden="true">&#9671;</span><span class="chat-nav-title">New chat</span>
      </button>
    </div>
  </nav>
</aside>
<header>
  <button id="workspaceMoreButton" title="More" aria-label="More" aria-expanded="false"
          onclick="toggleMenu()">&#8943;</button>
  <div class="header-menu" hidden>
    <button onclick="openEmptyChat()">New chat</button>
  </div>
</header>
<main class="workspace" data-chat-id="old-chat">
  <section class="conversation"><div class="message-row user">An earlier turn</div></section>
  <textarea id="messageInput"></textarea>
</main>
<script>
  window.clicks = [];
  window.creations = 0;
  function toggleMenu() {
    const menu = document.querySelector('.header-menu');
    menu.hidden = !menu.hidden;
    document.getElementById('workspaceMoreButton').setAttribute('aria-expanded', String(!menu.hidden));
  }
  async function openEmptyChat() {
    window.clicks.push('menu');
    document.querySelector('.header-menu').hidden = true;
    document.getElementById('workspaceMoreButton').setAttribute('aria-expanded', 'false');
    if (document.querySelectorAll('.message-row').length === 0) return;
    window.creations += 1;
    const response = await fetch('/api/v1/chats', {method: 'POST'});
    const payload = await response.json();
    if (!response.ok) return;
    document.querySelectorAll('.message-row').forEach(row => row.remove());
    // Empty messages precede the selected identity: waiting for detachment
    // alone would permit a job to be submitted to the old chat.
    setTimeout(() => {
      if (!window.freezeSelection) {
        document.querySelector('main.workspace').dataset.chatId = payload.chat?.chat_id || '';
      }
    }, 50);
  }
</script>
"""


@pytest.fixture
def workspace_page():
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        chromium = playwright.chromium.launch(headless=True)
        try:
            page = chromium.new_page()
            def response(route):
                if route.request.url.endswith('/api/v1/chats'):
                    route.fulfill(status=201, content_type='application/json',
                                  body=json.dumps({'chat': {'chat_id': 'created-chat'}}))
                else:
                    route.fulfill(content_type='text/html', body=WORKSPACE_WITH_A_SAVED_CHAT_NAMED_NEW_CHAT)
            page.route('http://fixture.invalid/**', response)
            page.goto('http://fixture.invalid/')
            yield page
        finally:
            chromium.close()


def test_new_chat_uses_the_menu_command_not_a_saved_chat_with_the_same_name(workspace_page):
    chat_id = browser.new_chat(workspace_page, 2000)

    assert chat_id == 'created-chat'
    assert workspace_page.evaluate("window.clicks") == ["menu"]
    assert workspace_page.locator('main.workspace').get_attribute('data-chat-id') == chat_id
    assert workspace_page.locator(".message-row").count() == 0


def test_new_chat_waits_for_the_workspace_to_select_its_first_chat(workspace_page):
    # Right after sign-in the chat list is still loading and no chat is
    # selected; the workspace selects the newest chat once the list arrives.
    workspace_page.evaluate("""() => {
        const workspace = document.querySelector('main.workspace');
        workspace.dataset.chatId = '';
        setTimeout(() => { workspace.dataset.chatId = 'old-chat'; }, 200);
    }""")

    assert browser.new_chat(workspace_page, 2000) == 'created-chat'


def test_new_chat_refuses_a_workspace_that_never_selects_a_chat(workspace_page):
    workspace_page.evaluate("document.querySelector('main.workspace').dataset.chatId = ''")

    with pytest.raises(AssertionError, match='no selected chat identity'):
        browser.new_chat(workspace_page, 300)
    assert workspace_page.evaluate('window.creations') == 0


def test_new_chat_keeps_an_already_open_menu_open(workspace_page):
    # A case that failed after opening the menu leaves it open; toggling it
    # closed again would leave only the saved chat to click.
    workspace_page.get_by_role("button", name="More", exact=True).click()

    browser.new_chat(workspace_page, 2000)

    assert workspace_page.evaluate("window.clicks") == ["menu"]
    assert workspace_page.locator(".message-row").count() == 0


def test_new_chat_reuses_an_already_empty_chat_without_creating_another(workspace_page):
    workspace_page.locator('.message-row').evaluate_all('(rows) => rows.forEach(row => row.remove())')

    assert browser.new_chat(workspace_page, 2000) == 'old-chat'
    assert workspace_page.evaluate('window.creations') == 0


@pytest.mark.parametrize('payload', [{}, {'chat': {}}, {'chat': {'chat_id': ''}},
                                    {'chat': {'chat_id': 'old-chat'}}])
def test_new_chat_refuses_malformed_or_reused_creation_receipts(workspace_page, payload):
    workspace_page.route('**/api/v1/chats', lambda route: route.fulfill(
        status=201, content_type='application/json', body=json.dumps(payload)))

    with pytest.raises(AssertionError, match='invalid new chat identity'):
        browser.new_chat(workspace_page, 2000)


def test_new_chat_does_not_treat_a_failed_create_as_an_empty_chat(workspace_page):
    workspace_page.route('**/api/v1/chats', lambda route: route.fulfill(
        status=503, content_type='application/json', body='{"error":"uncertain"}'))

    with pytest.raises(RuntimeError, match='HTTP 503'):
        browser.new_chat(workspace_page, 2000)
    assert workspace_page.locator('main.workspace').get_attribute('data-chat-id') == 'old-chat'


def test_detached_messages_do_not_prove_the_created_chat_was_selected(workspace_page):
    from playwright.sync_api import TimeoutError
    workspace_page.evaluate('window.freezeSelection = true')

    with pytest.raises(TimeoutError):
        browser.new_chat(workspace_page, 400)
    assert workspace_page.locator('.message-row').count() == 0
    assert workspace_page.locator('main.workspace').get_attribute('data-chat-id') == 'old-chat'


@pytest.fixture
def config_server():
    state = {'body': '', 'status': 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            config = self.path.split('?')[0].endswith('/config.js')
            body = state['body'] if config else '<script src="./config.js?v=fixture"></script><textarea id="messageInput"></textarea>'
            data = body.encode()
            self.send_response(state['status'] if config else 200)
            self.send_header('Content-Type', 'application/javascript' if config else 'text/html')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize('setting', [None, False])
def test_required_operations_is_an_acceptance_page_only_opt_in(config_server, setting):
    from playwright.sync_api import sync_playwright
    origin, state = config_server
    config = {'backendBaseUrl': 'http://backend.invalid', 'pollIntervalMs': 1000,
              'futureSetting': {'nested': [1, 2], 'enabled': False}}
    if setting is not None:
        config['requiredOperations'] = setting
    original = 'window.AGENTIC_CONFIG = Object.freeze(' + json.dumps(config) + ');'
    state['body'] = original

    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(headless=True)
        try:
            acceptance = instance.new_page()
            browser.sign_in(acceptance, origin, 'fixture@example.test', 'fixture-only', 2000)
            assert acceptance.evaluate('window.AGENTIC_CONFIG') == {**config, 'requiredOperations': True}
            assert acceptance.evaluate('Object.isFrozen(window.AGENTIC_CONFIG)') is True

            normal = instance.new_page()
            normal.goto(origin)
            assert normal.evaluate('window.AGENTIC_CONFIG') == config
            assert state['body'] == original

            # A config script outside the declared frontend path is untouched.
            other = acceptance.evaluate("async () => (await fetch('/other/config.js')).text()")
            assert other == original
        finally:
            instance.close()


@pytest.mark.parametrize('status', [404, 503])
def test_acceptance_config_override_preserves_upstream_failures(config_server, status):
    from playwright.sync_api import sync_playwright
    origin, state = config_server
    state.update(status=status, body='/* upstream configuration unavailable */')

    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(headless=True)
        try:
            page = instance.new_page()
            browser.enable_required_operations(page, origin)
            page.goto(origin)
            observed = page.evaluate("async () => { const r = await fetch('/config.js'); return {status:r.status, body:await r.text()}; }")
            assert observed == {'status': status, 'body': state['body']}
            assert page.evaluate('typeof window.AGENTIC_CONFIG') == 'undefined'
        finally:
            instance.close()
