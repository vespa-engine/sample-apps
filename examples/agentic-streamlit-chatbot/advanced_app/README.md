# Prerequisites

- Clone this repo and navigate to this directory.
- You need a python virtual environment. For this demo, `Python 3.13.1` was used, but any Python environment 3.11+ should work.
- You will need [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html) that you can deploy on MacOS with `brew install vespa-cli`
- Libraries dependencies can be installed with: `pip install -R requirements.txt`
- Sign-up with [Tavily](https://tavily.com/) and Get an API key.
- Spin-up a Vespa Cloud [Trial](https://vespa.ai/free-trial) account:
  - Login to the account you just created and create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/).
  - Save the tenant name.
- A Valid [OpenAI](https://openai.com/index/openai-api/) API key. Note that you have the option to use any other LLM, which may support Langgraph tools binding.
- Uncompress the data file: `unzstd data/*.zst`
  


# Deploy the Vespa Application


- Navigate to the `app` subdirectory
- Choose a name for your app. For example `ecommercebot`
- Follow instructions in the [**Getting Started**](https://cloud.vespa.ai/en/getting-started) document. Please note the following as you go through the documentation:
  - Skip Step 5. as you have already cloned the app directory.
  - You will need your tenant name you created previously.
  - When adding the public certificates with `vespa auth cert`, it will give you the absolute path of the certificate and the private key. Please note them down.
  - Return to the parent directory. To feed the application, run:
  ```
  cd ..
  vespa feed data/* --progress 30
  ```
- You can test the following query from the Vespa CLI:
  ```
  vespa query "select id, category, title, price, description  from sources * where default contains 'screwdriver'"
  ```
 - You will need the URL of your MTL endpoint for you Vespa application. Run the following command:
  ```
  vespa status
  ```
  This should return you an output like:
  ```
  Container container at https://xxxxx.yyyyy.z.vespa-app.cloud/ is ready
  ```
  Note down the URL for the MTLS end-point.

  # Configure and Launch your Streamlit Application
  
  A template for `secrets.toml` file to store streamlit secrets has been provided. Please create a subdirectory `.streamlit` under `vespa_shopping_assistant`  and copy the template there. 
  
  Update all the fields with all the information collected previously including the certicate locations and save the file as `secrets.toml`

  Optional. Launch random real-time updates for inventory:
  ```
  cd vespa_shopping_assistant
  python vespa_module/update_vespa.py
  ```
  
  Launch your streamlit application from `vespa_shopping_assistant`:
  ```
  streamlit run streamlit_app.py
  ```
You can now start asking some conversational questions, like for example looking for a glue, or rain gear.
