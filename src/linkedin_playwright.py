import pathlib
from typing import Optional
from rich.console import Console

CONTEXT_PATH = pathlib.Path.home() / ".linkedin_context.json"

async def async_linkedin_lookup(email: str, display_name: Optional[str] = None, debug: bool = False) -> Optional[dict]:
    from playwright.async_api import async_playwright
    import urllib.parse
    import os
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    profile_data = {"email": email, "data": {}}
    console = Console()

    if debug:
        console.print(f"[magenta]Launching Playwright browser for LinkedIn lookup: {email} ({display_name})")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        if CONTEXT_PATH.exists():
            if debug:
                console.print(f"[magenta]Loading LinkedIn browser context from {CONTEXT_PATH}")
            context = await browser.new_context(storage_state=str(CONTEXT_PATH))
        else:
            context = await browser.new_context()
        page = await context.new_page()

        # 1. Go directly to the search page
        if not display_name:
            await browser.close()
            return None
        if debug:
            console.print(f"[magenta]Searching LinkedIn by name: {display_name}")
        search_url = f'https://www.linkedin.com/search/results/people/?keywords={urllib.parse.quote(display_name)}'
        await page.goto(search_url, timeout=60000)
        await page.wait_for_timeout(3000)

        # 2. If redirected to login/checkpoint, prompt user to log in interactively
        if 'login' in page.url or 'checkpoint' in page.url or not page.url.startswith('https://www.linkedin.com/search/results/people'):
            console.print("[bold yellow]Please log in to LinkedIn in the opened browser window. Press Enter here when you have completed login.")
            input()
            await page.goto(search_url, timeout=60000)
            await page.wait_for_timeout(3000)
            if 'login' in page.url or 'checkpoint' in page.url or not page.url.startswith('https://www.linkedin.com/search/results/people'):
                console.print("[red]Login failed or not completed. Please try again.")
                await browser.close()
                return None
            await context.storage_state(path=str(CONTEXT_PATH))
            if debug:
                console.print(f"[green]Saved LinkedIn browser context to {CONTEXT_PATH}")

        # 3. Extract visible text of the results page
        results_text = await page.content()
        if debug:
            console.print(f"[magenta]Extracted search results page text (first 1000 chars):\n{results_text[:1000]}")
        
        # 4. Find and click the first search result using multiple strategies
        # Wait for search results to load
        search_results_url = page.url  # Store the search results page URL
        try:
            await page.wait_for_selector('a[href*="/in/"]', timeout=15000)
            console.print(f"[green]Found search results page")
        except Exception as e:
            if debug:
                console.print(f"[red]Timeout waiting for search results: {e}")
            await browser.close()
            return None
        
        profile_url = None  # Will store the actual profile URL
        profile_found = False
        profile_name = None
        
        # Strategy 1: Try direct selector for profile links
        try:
            profile_links = await page.query_selector_all('a[href*="/in/"]')
            if profile_links:
                first_link = profile_links[0]
                # Verify it's visible and clickable
                is_visible = await first_link.is_visible()
                if is_visible:
                    if debug:
                        console.print(f"[magenta]First link details:")
                        console.print(f"[dim]Tag name: {await first_link.evaluate('el => el.tagName')}")
                        console.print(f"[dim]Href: {await first_link.get_attribute('href')}")
                        console.print(f"[dim]Class: {await first_link.get_attribute('class')}")
                        console.print(f"[dim]Text content: {await first_link.inner_text()}")
                        console.print(f"[dim]Is visible: {is_visible}")
                        console.print(f"[dim]Bounding box: {await first_link.bounding_box()}")
                        console.print(f"[dim]Computed style: {await first_link.evaluate('el => window.getComputedStyle(el).display')}")
                    first_name_text = await first_link.get_attribute('aria-label') or await first_link.inner_text()
                    profile_url = await first_link.get_attribute('href')  # Store the profile URL
                    if debug:
                        console.print(f"[magenta]Found profile link: {first_link.text_content()}")
                        console.print(f"[magenta]Profile URL: {profile_url}")
                    await first_link.click()
                    await page.wait_for_timeout(4000)
                    profile_found = True
                    profile_name = first_name_text
        except Exception as e:
            if debug:
                console.print(f"[yellow]Direct selector failed: {e}")
        
        # Strategy 2: Use accessibility tree to find profile links (only if Strategy 1 failed)
        if not profile_found:
            try:
                acc_tree = await page.accessibility.snapshot(root=None, interesting_only=True)
                
                def find_profile_link(node):
                    if node is None:
                        return None
                    if node.get('role') == 'link' and node.get('name'):
                        # Check if this looks like a profile link
                        if any(keyword in node.get('name', '').lower() for keyword in ['profile', 'view profile', 'connect']):
                            return node
                    for child in node.get('children', []):
                        result = find_profile_link(child)
                        if result:
                            return result
                    return None
                
                profile_node = find_profile_link(acc_tree)
                if profile_node:
                    # Try to click using the accessibility node
                    await page.click(f'[aria-label*="{profile_node["name"]}"]', timeout=5000)
                    await page.wait_for_timeout(4000)
                    profile_found = True
                    profile_name = profile_node["name"]
            except Exception as e:
                if debug:
                    console.print(f"[yellow]Accessibility approach failed: {e}")
        
        # Strategy 3: Fallback - try to find any clickable element with name-like text (only if previous strategies failed)
        if not profile_found:
            try:
                # Look for elements that might be profile names
                name_elements = await page.query_selector_all('h3, h4, .name, [data-test-id*="name"]')
                for element in name_elements:
                    try:
                        text = await element.inner_text()
                        if text and len(text.strip()) > 2:  # Reasonable name length
                            # Try to find a clickable parent or sibling
                            clickable = await element.query_selector('a') or await element.evaluate_handle('el => el.closest("a")')
                            if clickable:
                                await clickable.click()
                                await page.wait_for_timeout(4000)
                                profile_found = True
                                profile_name = text.strip()
                                break
                    except Exception:
                        continue
            except Exception as e:
                if debug:
                    console.print(f"[yellow]Fallback approach failed: {e}")
        
        # If all strategies fail
        if not profile_found:
            if debug:
                console.print(f"[red]All strategies failed to find a clickable profile link")
            await browser.close()
            return None

        # 5. Extract only the readable profile information, stripping out any HTML.
        # Use Playwright's accessibility snapshot to get the accessibility tree and visible text.
        if debug:
            console.print(f"[magenta]Attempting to extract profile page text using multiple methods...")
        
        profile_text = ""
        
        # Method 1: Try accessibility snapshot first
        try:
            if debug:
                console.print(f"[yellow]Method 1: Using accessibility snapshot...")
            acc_tree = await page.accessibility.snapshot(root=None, interesting_only=True)
            def extract_text_from_acc(node):
                texts = []
                if node is None:
                    return texts
                if 'name' in node and node['name']:
                    texts.append(node['name'])
                for child in node.get('children', []):
                    texts.extend(extract_text_from_acc(child))
                return texts
            acc_texts = extract_text_from_acc(acc_tree)
            profile_text = "\n".join(acc_texts).strip()
            if debug:
                console.print(f"[green]Accessibility method extracted {len(profile_text)} characters")
        except Exception as e:
            if debug:
                console.print(f"[red]Accessibility method failed: {e}")
        
        # Method 2: Try inner_text on body if accessibility failed or returned little content
        if not profile_text or len(profile_text) < 100:
            try:
                if debug:
                    console.print(f"[yellow]Method 2: Using page.inner_text('body')...")
                profile_text = await page.inner_text('body')
                if debug:
                    console.print(f"[green]inner_text method extracted {len(profile_text)} characters")
            except Exception as e:
                if debug:
                    console.print(f"[red]inner_text method failed: {e}")
        
        # Method 3: Try evaluating JavaScript to get text content
        if not profile_text or len(profile_text) < 100:
            try:
                if debug:
                    console.print(f"[yellow]Method 3: Using JavaScript text extraction...")
                profile_text = await page.evaluate("""
                    () => {
                        const walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_TEXT,
                            null,
                            false
                        );
                        const texts = [];
                        let node;
                        while (node = walker.nextNode()) {
                            const text = node.textContent.trim();
                            if (text && text.length > 2) {
                                texts.push(text);
                            }
                        }
                        return texts.join('\\n');
                    }
                """)
                if debug:
                    console.print(f"[green]JavaScript method extracted {len(profile_text)} characters")
            except Exception as e:
                if debug:
                    console.print(f"[red]JavaScript method failed: {e}")
        
        # Method 4: Try specific LinkedIn selectors
        if not profile_text or len(profile_text) < 100:
            try:
                if debug:
                    console.print(f"[yellow]Method 4: Using LinkedIn-specific selectors...")
                selectors = [
                    '[data-section="summary"]',
                    '[data-section="experience"]',
                    '.pv-top-card-section__summary',
                    '.experience__section',
                    '.background-section',
                    'h1, h2, h3, h4, h5',
                    '.pv-text-details__left-panel',
                    '.pv-text-details__right-panel'
                ]
                texts = []
                for selector in selectors:
                    try:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            text = await element.inner_text()
                            if text and len(text.strip()) > 5:
                                texts.append(text.strip())
                    except Exception:
                        continue
                profile_text = "\n".join(texts).strip()
                if debug:
                    console.print(f"[green]LinkedIn selectors method extracted {len(profile_text)} characters")
            except Exception as e:
                if debug:
                    console.print(f"[red]LinkedIn selectors method failed: {e}")
        
        # Final fallback: get any text we can find
        if not profile_text:
            try:
                if debug:
                    console.print(f"[yellow]Final fallback: Getting any available text...")
                profile_text = await page.evaluate("document.body ? document.body.textContent || '' : ''")
                if debug:
                    console.print(f"[green]Fallback method extracted {len(profile_text)} characters")
            except Exception as e:
                if debug:
                    console.print(f"[red]Fallback method failed: {e}")
                profile_text = "Unable to extract profile text"
        
        if debug:
            console.print(f"[magenta]Final extracted profile text length: {len(profile_text)} characters")
            console.print(f"[magenta]First 1000 characters:\n{profile_text[:1000]}")
            if len(profile_text) < 100:
                console.print(f"[red]Warning: Very little text extracted ({len(profile_text)} chars)")

        # 6. Feed the profile text to OpenAI for a 2-line summary
        prompt_summary = (
            f"Summarize the following LinkedIn profile in 2 lines, focusing on the person's professional background and current role.\n\n"
            f"---\n{profile_text[:6000]}\n---"
        )
        if debug:
            console.print(f"[magenta]OpenAI prompt for profile summary:\n{prompt_summary[:1000]}")
        summary_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_summary}],
            max_tokens=200,
            temperature=0.2,
        )
        summary = summary_resp.choices[0].message.content.strip()
        if debug:
            console.print(f"[green]OpenAI profile summary:\n{summary}")

        await browser.close()
        return {"email": email, "data": {"summary": summary, "full_name": profile_name, "profile_url": profile_url, "search_results_url": search_results_url}} 