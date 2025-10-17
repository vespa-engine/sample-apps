<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Workshop - Ecommerce

## Dataset 

The dataset used in this workshop is this [Instacart dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) containing product and order data from an online grocery shopping service.

We add only a sample of the dataset in the github repository, but you can download (and preprocess) the full dataset to Vespa-compatible format from these links:

- https://data.vespa-cloud.com/sample-apps-data/workshop-ecommerce/orders_train.jsonl
- https://data.vespa-cloud.com/sample-apps-data/workshop-ecommerce/products.jsonl
- https://data.vespa-cloud.com/sample-apps-data/workshop-ecommerce/users.jsonl

The size of the dataset is 49688 products, 131209 orders and 231209 users.

We recommend using the sample data for quick iteration, and feeding the full dataset only after you are happy with your application.

If/when you want to use the full dataset, replace the sample files in the `dataset/` folder with the full dataset files above.

## 🚀 Quick Start

We have prepared a Vespa Cloud instance for this workshop, so you can get started quickly without having to deploy your own application. This application has been fed with the full dataset, and is ready to query.
Note that you will not be able to modify or feed additional data to this application.

For those of you want to build something off the pre-provisioned Vespa Cloud instance, follow these steps:

### 1. Configure Vespa CLI

Run:

```bash
vespa config set target cloud
vespa config set application vespa-team.workshop.default
vespa config set zone prod.aws-eu-west-1a
```

You can verify correct configuration with:

```bash
vespa config get
```

which should output:

```bash
application = vespa-team.workshop.default
cluster = <unset>
color = auto
debug = false
instance = default
quiet = false
target = cloud
zone = prod.aws-eu-west-1a
```

### 2. Set your `VESPA_WORKSHOP_TOKEN` environment variable (provided during the workshop):

```bash
export VESPA_WORKSHOP_TOKEN="<your-token-here>"
export VESPA_URL="<your-vespa-url-here>"
```

### 4. Test with a dummy query

```bash
vespa query -t $VESPA_URL \
  --header="Authorization: Bearer $VESPA_WORKSHOP_TOKEN" \
  'yql=select * from product.product where true limit 1'
```

### Alternatively, use pyvespa

We recommend using [uv](https://docs.astral.sh/uv/) for python project management.
After installing uv, create a virtual environment and install pyvespa:

```bash
uv venv
```

Activate the virtual environment:

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Windows (cmd):

```powershell
.venv\Scripts\activate.bat
```

And sync the dependencies to your virtual environment:

```bash
uv sync
```

You can then run the sample python notebook provided in `query_examples.ipynb`.
There, you will also need to provide the endpoint URL, which will be provided during the workshop.

## Deploy your own app using Vespa Cloud

Follow these steps to deploy an application to the [dev zone](https://cloud.vespa.ai/en/reference/zones.html) in the Vespa Cloud. Find more details and tips in the [developer guide](https://cloud.vespa.ai/en/developer-guide).

**Alternative:** [Run Vespa locally using Docker](https://docs.vespa.ai/en/vespa-quick-start.html)
Note that if you want to deploy locally, you need to comment out the `<clients>` section in `services.xml` to disable authentication for local deployments.

---

### Prerequisites

- Linux, macOS or Windows 10 Pro on x86_64 or arm64
- For macOS:
  - [Homebrew](https://brew.sh/) to install the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download from [GitHub releases](https://github.com/vespa-engine/vespa/releases)

---

### 1️⃣ Create a tenant in the Vespa Cloud

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one).

---

### 2️⃣ Install the Vespa CLI

**macOS / Linux:**

```bash
brew install vespa-cli
```

**Windows:**

Download the Windows release `.zip` from the [Vespa GitHub releases](https://github.com/vespa-engine/vespa/releases), extract it and add the extracted folder (containing `bin/vespa.exe`) to your `PATH`.

---

### 3️⃣ Configure the Vespa client

```bash
vespa config set target cloud
vespa config set application tenant-name.myapp # Replace tenant-name with your tenant name and myapp with your application name
```

Use your tenant name from step 1. This guide uses `myapp` as application name and `default` as instance name.

---

### 4️⃣ Authorize Vespa Cloud access

```bash
vespa auth login
```

Follow the instructions from the command to authenticate.

---

### 5️⃣ Get the application package

Navigate to your application directory (this workshop):

```bash
cd app
```

---

### 6️⃣ Add public certificate

```bash
vespa auth cert
```

This creates a self-signed certificate for data plane access (reads and writes). 

---

### 7️⃣ Deploy the application

```bash
vespa deploy --wait 600
```

The first deployment will take a few minutes while nodes are provisioned. Subsequent deployments will be quicker.

**Note:** Deployments to `dev` are removed 7 days after your last deployment. You can extend the expiry time in the Vespa Console.

---

### 8️⃣ Download Dataset

```bash
curl -O https://data.vespa-cloud.com/sample-apps-data/workshop/products.json
```

---

### 9️⃣ Feed Data to Vespa

```bash
vespa feed dataset/products.jsonl
```

```bash
vespa feed dataset/orders.jsonl
```

```bash
vespa feed dataset/users.jsonl
```

or if you have preprocessed the full dataset:

```bash
vespa feed dataset/products.jsonl
```

```bash
vespa feed dataset/orders_train.jsonl
```

```bash
vespa feed dataset/users.jsonl
```

---

## Examples of Queries

We also provide a [Jupyter notebook](query_examples.ipynb) with examples of how to run queries using pyvespa.


## 🔍 Example Queries - Products 🍞🥛

### Text search only

```bash
vespa query \
  "yql=select * from product.product where userQuery()" \
  "query=chocolate" \
  "ranking.profile=product_search"
```

### Text search with filter

```bash
vespa query \
  "yql=select * from product.product where userQuery() AND product_id>2" \
  "query=chocolate" \
  "ranking.profile=product_search"
```

### Hybrid search (text + vector)

```bash
vespa query \
  "yql=select * from product.product where userQuery() OR ({targetHits: 10}nearestNeighbor(embedding, q_emb))" \
  "query=chocolate" \
  "input.query(q_emb)=embed(@query)" \
  "ranking.profile=product_search"
```

And expect

```json
{
    "root": {
        "id": "toplevel",
        "relevance": 1.0,
        "fields": {
            "totalCount": 12
        },
        "coverage": {
            "coverage": 100,
            "documents": 12,
            "full": true,
            "nodes": 1,
            "results": 2,
            "resultsFull": 2
        },
        "errors": [
            {
                "code": 4,
                "summary": "Invalid query parameter",
                "source": "product",
                "message": "schema 'order' does not contain requested rank profile 'product_search'"
            }
        ],
        "children": [
            {
                "id": "id:products:product::12",
                "relevance": 1.036650349934333,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::12",
                    "product_id": 12,
                    "product_name": "Chocolate Fudge Layer Cake",
                    "aisle": "frozen dessert",
                    "department": "frozen",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6547879663348204,
                        "nativeRank(product_name)": 0.38186238359951247,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::1",
                "relevance": 1.0337756598100343,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::1",
                    "product_id": 1,
                    "product_name": "Chocolate Sandwich Cookies",
                    "aisle": "cookies cakes",
                    "department": "snacks",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6519132762105218,
                        "nativeRank(product_name)": 0.38186238359951247,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::9",
                "relevance": 0.6292770535089394,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::9",
                    "product_id": 9,
                    "product_name": "Light Strawberry Blueberry Yogurt",
                    "aisle": "yogurt",
                    "department": "dairy eggs",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6292770535089394,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::4",
                "relevance": 0.6176805692308868,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::4",
                    "product_id": 4,
                    "product_name": "Smart Ones Classic Favorites Mini Rigatoni With Vodka Cream Sauce",
                    "aisle": "frozen meals",
                    "department": "frozen",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6176805692308868,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::6",
                "relevance": 0.6139998265652206,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::6",
                    "product_id": 6,
                    "product_name": "Dry Nose Oil",
                    "aisle": "cold flu allergy",
                    "department": "personal care",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6139998265652206,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::3",
                "relevance": 0.6132707386896427,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::3",
                    "product_id": 3,
                    "product_name": "Robust Golden Unsweetened Oolong Tea",
                    "aisle": "tea",
                    "department": "beverages",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6132707386896427,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::5",
                "relevance": 0.6087317980200726,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::5",
                    "product_id": 5,
                    "product_name": "Green Chile Anytime Sauce",
                    "aisle": "marinades meat preparation",
                    "department": "pantry",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6087317980200726,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::11",
                "relevance": 0.6081764734620292,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::11",
                    "product_id": 11,
                    "product_name": "Peach Mango Juice",
                    "aisle": "refrigerated",
                    "department": "beverages",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6081764734620292,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::2",
                "relevance": 0.6076679191650988,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::2",
                    "product_id": 2,
                    "product_name": "All-Seasons Salt",
                    "aisle": "spices seasonings",
                    "department": "pantry",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6076679191650988,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:products:product::7",
                "relevance": 0.6073942736128445,
                "source": "product",
                "fields": {
                    "sddocname": "product",
                    "documentid": "id:products:product::7",
                    "product_id": 7,
                    "product_name": "Pure Coconut Water With Orange",
                    "aisle": "juice nectars",
                    "department": "beverages",
                    "summaryfeatures": {
                        "closeness(field,embedding)": 0.6073942736128445,
                        "nativeRank(product_name)": 0.0,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            }
        ]
    }
}
```

## Example Queries - Users 👤📊

### Fetch user by ID

```bash
vespa query \
  "yql=select * from user where user_id=1"
```

And expect

```json
{
    "root": {
        "id": "toplevel",
        "relevance": 1.0,
        "fields": {
            "totalCount": 1
        },
        "coverage": {
            "coverage": 100,
            "documents": 10,
            "full": true,
            "nodes": 1,
            "results": 1,
            "resultsFull": 1
        },
        "children": [
            {
                "id": "index:product/0/c4ca42382844020f3bf67638",
                "relevance": 0.0017429193899782135,
                "source": "product",
                "fields": {
                    "user_id": 1,
                    "user_profile": {
                        "27845": 1,
                        "46149": 1,
                        "13032": 1,
                        "26088": 1,
                        "39657": 1,
                        "25133": 1,
                        "38928": 1,
                        "10258": 1,
                        "49235": 1,
                        "196": 1,
                        "26405": 1
                    },
                    "user_purchases": [
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 196
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 25133
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 38928
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 26405
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 39657
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 10258
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 13032
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 26088
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 27845
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 49235
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 46149
                        }
                    ]
                }
            }
        ]
    }
}
```

### Find users who have purchased specific products using WAND

See https://docs.vespa.ai/en/using-wand-with-vespa.html 

```bash
vespa query \
    'yql=select * from user where ({targetHits: 25}wand(user_profile, [[196,1]]))' \
    'ranking=unranked'
```

Or to find users who have purchased multiple products:

```bash
vespa query \
    'yql=select * from user where ({targetHits: 25}wand(user_profile, [[196,1],[25133,1]]))' \
    'ranking=unranked'
```

Or to have users ranked by the sparse dot product of their user profile and a query vector representing products they have purchased:

```bash
vespa query \
    'yql=select * from user where ({targetHits: 25}wand(user_profile, [[196,1],[25133,1]]))' \
    'ranking=sparse_dot'
```

Notice that relevance is now 2.0:

```json
{
    "root": {
        "id": "toplevel",
        "relevance": 1.0,
        "fields": {
            "totalCount": 1
        },
        "coverage": {
            "coverage": 100,
            "documents": 10,
            "full": true,
            "nodes": 1,
            "results": 1,
            "resultsFull": 1
        },
        "children": [
            {
                "id": "id:users:user::1",
                "relevance": 2.0,
                "source": "product",
                "fields": {
                    "sddocname": "user",
                    "documentid": "id:users:user::1",
                    "user_id": 1,
                    "user_profile": {
                        "27845": 1,
                        "46149": 1,
                        "13032": 1,
                        "26088": 1,
                        "39657": 1,
                        "25133": 1,
                        "38928": 1,
                        "10258": 1,
                        "49235": 1,
                        "196": 1,
                        "26405": 1
                    },
                    "user_purchases": [
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 196
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 25133
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 38928
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 26405
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 39657
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 10258
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 13032
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 26088
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 27845
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 49235
                        },
                        {
                            "order_dow": 4,
                            "order_hour_of_day": 8,
                            "order_id": 1187899,
                            "product_id": 46149
                        }
                    ]
                }
            }
        ]
    }
}
```

---

## Example Queries - Orders 🛒🌐

### Fetch order by ID

```bash
vespa query \
  "yql=select * from order where order_id=1187899"
```

### Find orders containing a specific product

```bash
vespa query \
  'yql=select * from order where product_ids contains "196"'
```

### Find orders containing multiple specific products

```bash
vespa query \
  'yql=select * from order where product_ids contains "196" AND product_ids contains "25133"'
```

And expect

```json
{
    "root": {
        "id": "toplevel",
        "relevance": 1.0,
        "fields": {
            "totalCount": 1
        },
        "coverage": {
            "coverage": 100,
            "documents": 12,
            "full": true,
            "nodes": 1,
            "results": 1,
            "resultsFull": 1
        },
        "children": [
            {
                "id": "id:orders:order::1187899",
                "relevance": 0.0017429193899782135,
                "source": "product",
                "fields": {
                    "sddocname": "order",
                    "documentid": "id:orders:order::1187899",
                    "order_id": 1187899,
                    "user_id": 1,
                    "order_dow": 4,
                    "order_hour_of_day": 8,
                    "days_since_prior_order": 14.0,
                    "product_ids": [
                        196,
                        25133,
                        38928,
                        26405,
                        39657,
                        10258,
                        13032,
                        26088,
                        27845,
                        49235,
                        46149
                    ]
                }
            }
        ]
    }
}
```

---

## Some ideas

- (Personalized) recommendations based on order history (and possibly time of day / day of week)
- Recommendations given a temporary basket of products
- Demo UI
- Search-as-you-type functionality
  
# Useful links

- [Vespa Documentation](https://docs.vespa.ai/)
- [Vespa Cloud](https://cloud.vespa.ai/)
- [Vespa Sample Applications](https://github.com/vespa-engine/sample-apps)
- [Pyvespa - Python client for Vespa](https://vespa-engine.github.io/pyvespa/)
- [Vespa Documentation Search](https://search.vespa.ai)
- [Tensor playground](https://docs.vespa.ai/playground)
- [Vespa slack community](https://slack.vespa.ai/)
