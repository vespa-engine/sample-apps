from fasthtml_hf import setup_hf_backup
from fasthtml.common import (
    picolink,
    serve,
    Div,
    Title,
    Main,
    Input,
    Button,
    A,
    Section,
    H2,
    Ul,
    Li,
    P,
    Img,
    Details,
    MarkdownJS,
    HighlightJS,
    Summary,
    Script,
    I,
    Form,
    RedirectResponse,
    dataclass,
    Favicon,
    database,
    get_key,
    Table,
    Thead,
    Tr,
    Th,
    Tbody,
    Td,
    FileResponse,
    fast_app,
    Beforeware,
    Hidden,
    Request,
    H3,
    Style,
)
from fasthtml.components import Nav, Article, Header, Mark
from fasthtml.pico import Search, Grid, Fieldset, Label
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from vespa.application import Vespa
import json
import os
import re
import time
from hmac import compare_digest
from io import StringIO
import csv
import tempfile
from enum import Enum
from typing import Tuple as T
from urllib.parse import quote

DEV_MODE = False

if DEV_MODE:
    print("Running in DEV_MODE - Hot reload enabled")
    print("Loading environment variables from .env")
    from dotenv import load_dotenv

    load_dotenv()
else:
    print("DEV_MODE disabled - environment variables loaded from system")

vespa_app_url = os.getenv("VESPA_APP_URL", None)
if vespa_app_url is None:
    print("Please set the VESPA_APP_URL environment variable")
    exit(1)

ADMIN_NAME = os.getenv("ADMIN_NAME", "admin")
ADMIN_PWD = os.getenv("ADMIN_PWD", "admin")

vespa_app: Vespa = Vespa(
    url=vespa_app_url,
    vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"),
)
status = vespa_app.get_application_status()
if status is None:
    print("Could not connect to Vespa application")
else:
    print("Connected to Vespa application!")

fa = Script(src="https://kit.fontawesome.com/664eb1a115.js", crossorigin="anonymous")
favicon = Favicon(
    "https://search.vespa.ai/favicon.ico",
    "https://search.vespa.ai/favicon.ico",
)
DB_FILE = "db/vespa.db"
db = database(DB_FILE)
queries = db.t.queries
if queries not in db.t:
    # You can pass a dict, or kwargs, to most MiniDataAPI methods.
    queries.create(
        dict(qid=int, query=str, ranking=str, sess_id=str, timestamp=int), pk="qid"
    )
    # Add autoincrement to the qid column
    db.query("ALTER TABLE queries ADD COLUMN qid INTEGER PRIMARY KEY AUTOINCREMENT")
Query = queries.dataclass()

# Add a classmethod to the Query dataclass to convert timestamp field to a human readable format
Query.get_datetime = lambda self: time.strftime(
    "%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)
)

# Status code 303 is a redirect that can change POST to GET,
# so it's appropriate for a login page.
login_redir = RedirectResponse("/login", status_code=303)


def user_auth_before(req, sess):
    # The `auth` key in the request scope is automatically provided
    # to any handler which requests it, and can not be injected
    # by the user using query params, cookies, etc, so it should
    # be secure to use.
    print(f"Session Data before route: {sess}")
    auth = req.scope["auth"] = sess.get("auth", None)
    print(f"Auth: {auth}")
    if not auth:
        return login_redir


spinner_css = Style("""
    .htmx-indicator {
        display: none; /* Hide spinner by default */
    }

    .htmx-indicator.htmx-request {
        display: block;    
    }
""")

headers = (
    picolink,
    MarkdownJS(),
    HighlightJS(langs=["json", "python"]),
    favicon,
    fa,
    spinner_css,
)

# Read file contents once before starting the server
with open("README.md") as f:
    README = f.read()
with open("main.py") as f:
    SOURCE = f.read()

# Sesskey
sess_key_path = "session/.sesskey"
# Make sure session directory exists
os.makedirs("session", exist_ok=True)


# Middleware
class XFrameOptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "ALLOW-FROM https://huggingface.co/"
        return response


middlewares = [
    Middleware(
        SessionMiddleware,
        secret_key=get_key(fname=sess_key_path),
        max_age=3600,
    ),
    Middleware(XFrameOptionsMiddleware),
]
bware = Beforeware(
    user_auth_before,
    skip=[
        r"/favicon\.ico",
        r"/static/.*",
        r".*\.css",
        r".*\.js",
        "/",
        "/login",
        "/search",
        "/document/.*",
        "/expand/.*",
        "/source",
        "/about",
    ],
)

app, rt = fast_app(
    before=bware,
    live=DEV_MODE,
    hdrs=headers,
    middleware=middlewares,
    key_fname=sess_key_path,
    same_site="None",
)


sesskey = get_key(fname=sess_key_path)
print(f"Session key: {sesskey}")


# enum class for rank profiles
class RankProfile(str, Enum):
    bm25 = "bm25"
    semantic = "semantic"
    fusion = "fusion"


def get_navbar(admin: bool):
    print(f"In get_navbar: {admin}")
    bar = Nav(
        Ul(
            Li(
                A(
                    Img(src="https://vespa.ai/assets/vespa-ai-logo-heather.svg"),
                    href="https://cloud.vespa.ai",
                    target="_blank",
                    style="margin: 10px;",
                ),
            )
        ),
        Ul(H2("Vespa-fastHTML demo")),
        Ul(
            # A question mark icon with link to an about page
            A(
                I(cls="fa fa-question-circle fa-2x"),
                href="/about",
                style="margin: 10px;",
                title="About this app",
            ),
            A(
                I(cls="fab fa-slack fa-2x"),
                href="https://slack.vespa.ai/",
                style="margin: 10px;",
                target="_blank",
                title="Join Vespa Slack channel",
            ),
            A(
                I(cls="fab fa-github fa-2x"),
                href="https://github.com/vespa-engine/sample-apps/tree/master/examples/fasthtml-demo",
                style="margin: 10px;",
                target="_blank",
                title="View source code on GitHub",
            ),
            A(
                I(cls="fa fa-code fa-2x"),
                href="/source",
                style="margin: 10px;",
                title="View source code",
            ),
            # Login icon (link to /login) show tooltip on hover. MAke it hidden if admin is logged in
            A(
                I(cls="fa fa-shield fa-2x"),
                href="/login" if not admin else "/admin",
                style="margin: 10px;",
                title="Admin login",
            ),
            # Logout icon if admin is logged in
            A(
                I(cls="fa fa-sign-out fa-2x"),
                href="/logout",
                style="margin: 10px;" if admin else "display: none;",
                title="Logout",
            ),
        ),
        # 10px margin to right of navbar
        style="margin-right: 10px;",
    )
    return bar


def spinner_div(hidden: bool = False):
    return Div(
        A(
            id="spinner",
            aria_busy="true",
            cls="htmx-indicator",
            style="font-size: 2em;",
        ),
        style="text-align: center; margin-top: 40px;"
        if not hidden
        else "display: none;",
    )


@app.route("/")
def get(sess):
    # Can not get auth directly, as it is skipped in beforeware
    auth = sess.get("auth", False)
    queries = [
        "Breast Cancer Cells Feed on Cholesterol",
        "Treating Asthma With Plants vs. Pills",
        "Alkylphenol Endocrine Disruptors",
        "Testing Turmeric on Smokers",
        "The Role of Pesticides in Parkinson's Disease",
    ]
    return (
        Title("Vespa demo"),
        get_navbar(auth),
        Main(
            # Search bar
            Search(
                Input(
                    type="search",
                    placeholder="Ask/search for medical information?",
                    id="userquery",
                ),
                # Get search results on button click with search-input as query parameter
                Button(
                    "Search",
                    hx_get="/search",
                    # include userquery and id of selected ranking radio button
                    hx_include="#userquery, input[name=ranking]:checked",
                    hx_target="#results",
                    hx_indicator="#spinner",
                ),
                style="margin: 10% 10px 0 0;",
            ),
            Fieldset(
                Input(type="radio", id="bm25", name="ranking", value="bm25"),
                Label("BM25", htmlfor="bm25"),
                Input(type="radio", id="semantic", name="ranking", value="semantic"),
                Label("Semantic", htmlfor="semantic"),
                Input(
                    type="radio",
                    id="fusion",
                    name="ranking",
                    value="fusion",
                    checked="",
                ),
                Label("Reciprocal Rank fusion", htmlfor="fusion"),
                style="margin: 10px; text-align: center;",
                id="ranking",
            ),
            H3("Example queries"),
            # Buttons with predefined search queries
            Grid(
                *[
                    Button(
                        query,
                        hx_get="/search?userquery=" + query,
                        hx_include="input[name=ranking]:checked",
                        hx_target="#results",
                        hx_indicator="#spinner",
                        hx_on_click=f"document.getElementById('userquery').value='{query}'",
                        style="margin: 10px; padding: 5px;",
                        cls="secondary outline",
                        id=f"example-{qid}",
                    )
                    for qid, query in enumerate(queries)
                ],
                # Make the grid buttons have same height and distribute evenly and center align
                style="grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));",
            ),
            # Section(
            #     Input(
            #         id="suggestion-input",
            #         list="search-options",
            #         placeholder="Search options",
            #     ),
            #     Datalist(
            #             *[
            #                 Option(
            #                     "Covid-19",
            #                     value="Covid-19",
            #                 ),
            #                 Option(
            #                     "Vaccine",
            #                     value="Vaccine",
            #                 ),
            #             ],
            #             id="search-options",
            #         ),
            #     id="suggestions",
            # ),
            # Display spinner div only if it #spinner does not exist
            Section(
                spinner_div(),
                id="results",
                hx_swap="innerHTML",
                style="margin: 20px;",
            ),
            style="margin: 0 auto; width: 70%;",
            id="main",
        ),
    )


@dataclass
class Login:
    name: str
    pwd: str


@app.get("/login")
def get_login_form(sess, error: bool = False):
    auth = sess.get("auth", False)
    frm = Form(
        Input(id="name", placeholder="Name"),
        Input(id="pwd", type="password", placeholder="Password"),
        Button("login"),
        action="/login",
        method="post",
    )
    err_msg = P("Incorrect password", style="color: red;") if error else ""
    return (
        Title("Admin login"),
        get_navbar(auth),
        Main(
            err_msg,
            frm,
            style="width: 50%; margin: 10% auto;",
        ),
    )


@app.post("/login")
def post(login: Login, sess):
    if not compare_digest(ADMIN_PWD.encode("utf-8"), login.pwd.encode("utf-8")):
        # Incorrect password - add error message
        return RedirectResponse("/login?error=True", status_code=303)
    sess["auth"] = True
    print(f"Sess after login: {sess}")
    return RedirectResponse("/admin", status_code=303)


@app.get("/logout")
def logout(sess):
    sess["auth"] = False
    return RedirectResponse("/")


def replace_hi_with_strong(text):
    parts = re.split(r"(<hi>|</hi>)", text)
    elements = []
    open_tag = False
    for part in parts:
        if part == "<hi>":
            open_tag = True
        elif part == "</hi>":
            open_tag = False
        elif open_tag:
            elements.append(Mark(part))
        else:
            elements.append(part)
    return elements


def log_query_to_db(query, ranking, sess):
    return queries.insert(
        Query(query=query, ranking=ranking, sess_id=sesskey, timestamp=int(time.time()))
    )


def parse_results(records):
    return [
        Article(
            Header(
                H2(
                    A(
                        result["title"],
                        hx_get=f"/document/{result['id']}",
                        hx_target="#results",
                    )
                )
            ),
            Div(
                P(
                    *replace_hi_with_strong(
                        result["body"][:300] + "..."
                    ),  # Display first 300 characters of body
                ),
                Div(
                    # Button with "Show more" - center align
                    Button(
                        "Show more",
                        hx_post=f"/expand/{result['id']}?expand=true",
                        hx_target=f"#{result['id']}",
                        hx_include=f"#{result['id']}-full",
                        cls="outline secondary",
                        # Style to fill whole width of parent div
                        style="width: 100%;",
                    ),
                    style="text-align: center;",
                ),
                id=result["id"],
            ),
            Hidden(result["body"], id=f"{result['id']}-full"),
        )
        for result in records
    ]


@app.post("/expand/{docid}")
async def expand(request: Request, docid: str, expand: bool):
    print(f"Expanding {docid}")
    form_data = await request.form()
    result = form_data.get(f"{docid}-full")
    if not expand:
        result = result[:300] + "..."
    return (
        Div(
            P(
                *replace_hi_with_strong(result),  # Display full body
            ),
            Div(
                # Button with "Show less" - center align
                Button(
                    "Show less" if expand else "Show more",
                    hx_post=f"/expand/{docid}?expand="
                    + ("false" if expand else "true"),
                    hx_target=f"#{docid}",
                    hx_include=f"#{docid}-full",
                    cls="outline secondary",
                    # Style to fill whole width of parent div
                    style="width: 100%;",
                ),
                style="text-align: center;",
            ),
            id=docid,
        ),
    )


# Returns tuple of (yql, body(dict)) based on the ranking profile
def get_yql(ranking: RankProfile, userquery: str) -> T[str, dict]:
    if ranking == RankProfile.bm25:
        yql = "select * from sources * where userQuery() limit 10"
        body = {}
    elif ranking == RankProfile.semantic:
        yql = "select * from sources * where ({targetHits:10}nearestNeighbor(embedding,q)) limit 10"
        body = {"input.query(q)": f"embed({userquery})"}
    elif ranking == RankProfile.fusion:
        yql = "select * from sources * where rank({targetHits:1000}nearestNeighbor(embedding,q), userQuery()) limit 10"
        body = {"input.query(q)": f"embed({userquery})"}
    return yql, body


@app.get("/search")
async def search(userquery: str, ranking: str, sess):
    print(sess)
    if "queries" not in sess:
        sess["queries"] = []
    quoted = quote(userquery) + "&ranking=" + ranking
    sess["queries"].append(quoted)
    print(f"Searching for: {userquery}")
    print(f"Ranking: {ranking}")
    log_query_to_db(userquery, ranking, sess)
    yql, body = get_yql(ranking, userquery)
    async with vespa_app.asyncio() as session:
        resp = await session.query(
            yql=yql,
            query=userquery,
            hits=10,
            ranking=str(ranking),
            body=body,
        )
    records = []
    fields = ["id", "title", "body"]
    for hit in resp.hits:
        record = {}
        for field in fields:
            record[field] = hit["fields"][field]
        records.append(record)
    results = parse_results(records)
    json_dump = json.dumps(resp.get_json(), indent=4)
    return Div(
        spinner_div(),
        # Accordion (with Details)
        Details(
            Summary("Full JSON response"),
            Div(
                f"""```json\n{json_dump}\n```""",
                cls="marked",
            ),
        ),
        H2(
            "Search Results",
        ),
        Div(
            *results,
            id="all-searchresults",
        ),
    )


@app.get("/download_csv")
def download_csv(auth):
    queries_dict = list(db.query("SELECT * FROM queries"))
    queries = [Query(**query) for query in queries_dict]

    # Create CSV in memory
    csv_file = StringIO()
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Query", "Session ID", "Timestamp"])
    for query in queries:
        csv_writer.writerow([query.query, query.sess_id, query.timestamp])

    # Move to the beginning of the StringIO object
    csv_file.seek(0)

    # Save CSV to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.write(csv_file.getvalue().encode("utf-8"))
    temp_file.close()

    return FileResponse(
        temp_file.name,
        filename="queries.csv",
        media_type="text/csv",
        content_disposition_type="attachment",
    )


@app.get("/admin")
def get_admin(auth, page: int = 1):
    limit = 15
    offset = (page - 1) * limit
    total_queries_result = list(
        db.query("SELECT COUNT(*) AS count FROM queries ORDER BY timestamp DESC")
    )
    total_queries = total_queries_result[0]["count"]
    queries_dict = list(
        db.query(f"SELECT * FROM queries LIMIT {limit} OFFSET {offset}")
    )
    queries = [Query(**query) for query in queries_dict]

    total_pages = (
        total_queries + limit - 1
    ) // limit  # Calculate total number of pages

    # Define the range of pages to display
    page_window = 5  # Number of pages to display at once
    start_page = max(1, page - page_window // 2)
    end_page = min(total_pages, start_page + page_window - 1)

    # Adjust the start and end pages if they exceed the limits
    if end_page - start_page < page_window:
        start_page = max(1, end_page - page_window + 1)

    # Pagination controls with "First", "Previous", "Next", and "Last"
    pagination_controls = Div(
        A(
            "First",
            href="/admin?page=1",
            style="margin: 5px;"
            if page > 1
            else "margin: 5px; color: grey; pointer-events: none;",
        ),
        A(
            "Previous",
            href=f"/admin?page={page - 1}",
            style="margin: 5px;"
            if page > 1
            else "margin: 5px; color: grey; pointer-events: none;",
        ),
        *[
            A(
                f"{i}",
                href=f"/admin?page={i}",
                style="margin: 5px;"
                if i != page
                else "margin: 5px; font-weight: bold;",
            )
            for i in range(start_page, end_page + 1)
        ],
        A(
            "Next",
            href=f"/admin?page={page + 1}",
            style="margin: 5px;"
            if page < total_pages
            else "margin: 5px; color: grey; pointer-events: none;",
        ),
        A(
            "Last",
            href=f"/admin?page={total_pages}",
            style="margin: 5px;"
            if page < total_pages
            else "margin: 5px; color: grey; pointer-events: none;",
        ),
        style="text-align: center; margin: 20px;",
    )

    # Total pages indication
    total_pages_indicator = Div(
        f"Page {page} of {total_pages}",
        style="text-align: center; margin: 10px;",
    )

    return (
        Title("Admin"),
        get_navbar(auth),
        Main(
            Div(
                A(
                    I(cls="fa fa-arrow-left"),
                    "Back",
                    href="/",
                    title="Back to main page",
                    style="margin: 10px;",
                ),
                style="margin: 10px;",
            ),
            H2("Queries"),
            # Table of all queries
            Table(
                Thead(
                    Tr(
                        Th("Query"),
                        Th("Session ID"),
                        Th("Datetime"),
                    )
                ),
                Tbody(
                    *[
                        Tr(
                            Td(query.query),
                            Td(query.sess_id),
                            Td(query.get_datetime()),
                        )
                        for query in queries
                    ],
                ),
                cls="striped",
            ),
            total_pages_indicator,  # Include the total pages indicator here
            pagination_controls,
            Div(
                A(
                    I(cls="fa fa-download fa-2x"),
                    " Download CSV",
                    href="/download_csv",
                    style="margin: 10px; float: right;",
                    title="Download queries as CSV",
                ),
                style="text-align: right; margin: 20px;",
            ),
            style="width: 80%; margin: 40px auto;",
        ),
    )


@app.get("/source")
def get_source(auth, sess):
    # Back icon to go back to main page in top left corner
    return (
        Title("Source code"),
        get_navbar(auth),
        Main(
            Div(
                A(
                    I(cls="fa fa-arrow-left"),
                    "Back",
                    href="/",
                    title="Back to main page",
                    style="margin: 10px;",
                ),
                Div(
                    f"""### `main.py`\n### This is the complete source code for this app \n```python\n{SOURCE}\n```""",
                    cls="marked",
                    style="margin: 10px;",
                ),
                style="width: 80%; margin: 40px auto;",
            ),
        ),
    )


@app.get("/about")
def get_about(auth, sess):
    # Strip everything before the FIRST # in the README
    stripped_readme = re.sub(
        r"^.*?(?=# FastHTML Vespa frontend)", "", README, flags=re.DOTALL
    )

    return (
        Title("About this app"),
        get_navbar(auth),
        Main(
            Div(
                A(
                    I(cls="fa fa-arrow-left"),
                    "Back",
                    href="/",
                    title="Back to main page",
                    style="margin: 10px;",
                ),
                Div(
                    stripped_readme,
                    cls="marked",
                    style="margin: 10px;",
                ),
                style="width: 80%; margin: 40px auto;",
            ),
        ),
    )


@app.get("/document/{docid}")
def get_document(docid: str, sess):
    resp = vespa_app.get_data(data_id=docid, schema="doc", namespace="tutorial")
    doc = resp.json
    # Link with Back to search results at top of page
    return Main(
        Div(
            A(
                I(cls="fa fa-arrow-left"),
                "Back to search results",
                hx_get=f"/search?userquery={sess['queries'][-1]}",
                hx_target="#results",
                style="margin: 10px;",
            ),
            H2(doc["fields"]["title"], style="margin: 10px;"),
            P(doc["fields"]["body"], cls="marked"),
        ),
    )


if not DEV_MODE:
    try:
        setup_hf_backup(app)
    except Exception as e:
        print(f"Error setting up hf backup: {e}")
serve()
