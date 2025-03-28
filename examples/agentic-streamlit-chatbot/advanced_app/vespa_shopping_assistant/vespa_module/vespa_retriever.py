from vespa.application import Vespa
import vespa.querybuilder as qb
from vespa.querybuilder import QueryField as qf
from utils.secrets import VESPA_URL, PUBLIC_CERT_PATH, PRIVATE_KEY_PATH
from models.state import SubgraphState


def create_conditions(conditions_list):
    """Convert an array of condition strings into Vespa QueryBuilder conditions."""

    # Define supported operators
    operators = {
        ">": lambda f, v: f > v,
        "<": lambda f, v: f < v,
        "=": lambda f, v: f == v,
        "!=": lambda f, v: f != v,
        ">=": lambda f, v: f >= v,
        "<=": lambda f, v: f <= v,
        "contains": lambda f, v: f.contains(v),  # Special case for text search
    }

    parsed_conditions = []

    for condition_str in conditions_list:
        parts = condition_str.split(
            " ", 2
        )  # Split into max 3 parts to handle "contains" operator properly

        if len(parts) < 3:
            raise ValueError(
                f"Invalid condition format: '{condition_str}'. Expected format: 'field operator value'"
            )

        field, operator, value = parts[0], parts[1], parts[2]

        # Convert numeric values if applicable
        try:
            value = float(value) if "." in value or value.isdigit() else value
        except ValueError:
            pass  # Keep as string if conversion fails

        # Validate operator
        if operator not in operators:
            raise ValueError(
                f"Unsupported operator: '{operator}' in condition '{condition_str}'"
            )

        # Create QueryField object and apply operator function
        query_field = qf(field)
        parsed_conditions.append(operators[operator](query_field, value))

    return parsed_conditions  # Returns a list of Condition objects


def VespaRetriever(state: SubgraphState):
    user_query = state["SearchEngineQuery"]
    filters = state["Filters"]
    num_hits = 10

    print(filters)
    columns = ["id", "category", "title", "description", "average_rating", "price"]
    schema = "product"

    inlistcol = qf("category")
    catlist = state["Categories"]

    vespa_app = Vespa(url=VESPA_URL, cert=PUBLIC_CERT_PATH, key=PRIVATE_KEY_PATH)

    condition_uq = qb.userQuery(user_query)

    condition_ann = qb.nearestNeighbor(
        field="embedding",
        query_vector="q_embedding",
    )

    condition_cat = inlistcol.in_(*catlist)

    if filters != ["None"]:
        cond_list = create_conditions(filters)
        cond_opt = qb.all(*cond_list)

        q = (
            qb.select(columns)
            .from_(schema)
            .where((condition_ann | condition_uq) & condition_cat & cond_opt)
        )
    else:
        q = (
            qb.select(columns)
            .from_(schema)
            .where((condition_ann | condition_uq) & condition_cat)
        )

    print(str(q))
    resp = vespa_app.query(
        yql=q,
        ranking="hybrid",
        body={"input.query(q_embedding)": f"embed({user_query})"},
        hits=num_hits,
    )

    # Extract only the 'fields' content from each entry
    filtered_data = [hit["fields"] for hit in resp.hits]

    state["SearchEngineResults"] = filtered_data
    print(state["SearchEngineResults"])

    return state
