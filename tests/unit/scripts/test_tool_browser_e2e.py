"""Deterministic browser observation waits; no browser or provider is started."""
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
<section class="conversation"><div class="message-row user">An earlier turn</div></section>
<textarea id="messageInput"></textarea>
<script>
  window.clicks = [];
  function toggleMenu() {
    const menu = document.querySelector('.header-menu');
    menu.hidden = !menu.hidden;
    document.getElementById('workspaceMoreButton').setAttribute('aria-expanded', String(!menu.hidden));
  }
  function openEmptyChat() {
    window.clicks.push('menu');
    setTimeout(() => document.querySelectorAll('.message-row').forEach(row => row.remove()), 50);
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
            page.set_content(WORKSPACE_WITH_A_SAVED_CHAT_NAMED_NEW_CHAT)
            yield page
        finally:
            chromium.close()


def test_new_chat_uses_the_menu_command_not_a_saved_chat_with_the_same_name(workspace_page):
    browser.new_chat(workspace_page, 2000)

    assert workspace_page.evaluate("window.clicks") == ["menu"]
    assert workspace_page.locator(".message-row").count() == 0


def test_new_chat_keeps_an_already_open_menu_open(workspace_page):
    # A case that failed after opening the menu leaves it open; toggling it
    # closed again would leave only the saved chat to click.
    workspace_page.get_by_role("button", name="More", exact=True).click()

    browser.new_chat(workspace_page, 2000)

    assert workspace_page.evaluate("window.clicks") == ["menu"]
    assert workspace_page.locator(".message-row").count() == 0
