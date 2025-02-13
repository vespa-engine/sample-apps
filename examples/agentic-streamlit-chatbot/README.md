# Prerequisites


- You need a python virtual environment. For this demo, `Python 3.13.1` was used, but any Python environment 3.11+ should work.
- You will need [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html) that you can deploy on MacOS with `brew install vespa-cli`
- Libraries dependencies can be installed with: `pip install -R requirements.txt`
- Sign-up with [Tavily](https://tavily.com/) and Get an API key.
- Spin-up a Vespa Cloud [Trial](https://vespa.ai/free-trial) account:
  - Login to the account you just created and create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/).
  - Save the tenant name.
- A Valid [OpenAI](https://platform.openai.com/docs/api-reference/introduction) API key. Note that you have the option to use any other LLM, which may support Langgraph tools binding.
- Git clone the repo `https://github.com/vespa-engine/system-test.git`
- The ecommerce_hybrid_search app will be deployed. For more information about the app, please review the [README.md](https://github.com/vespa-engine/system-test/blob/master/tests/performance/ecommerce_hybrid_search/dataprep/README.md). You do not have to follow the data prep steps there. Follow the instructions below instead.
- Uncompress the data file: `zstd -d data/vespa_update-96k.json.zst`
  


# Deploy the Ecommerce Vespa Application


- In the system-test repo you just cloned, navigate to `tests/performance/ecommerce_hybrid_search/app`
- Choose a name for your app. For example `ecommercebot`
- Follow instructions in the [**Getting Started**](https://cloud.vespa.ai/en/getting-started) document. Please note the following as you go through the documentation:
  - You will need your tenant name you created previously.
  - When adding the public certificates with `vespa auth cert`, it will give you the absolute path of the certificate and the private key. Please note them down.
  - To feed the application, return to the original directory and run:
  ```
  vespa feed data/vespa_update-96k.json
  ```
- You can test the following query from the Vespa CLI:
  ```
  vespa query "select id, category, title, price  from sources * where default contains 'screwdriver'"
  ```
 - You will need the URL of your Vespa application. Run the following command:
  ```
  vespa status
  ```
  This should return you an output like:
  ```
  Container container at https://xxxxx.yyyyy.z.vespa-app.cloud/ is ready
  ```
  Note down the URL.

  # Configure and Launch your Streamlit Application

  A template for `secrets.toml` file to store streamlit secrets has been provided. Please create a subdirectory `.streamlit` and copy the template there. 
  
  Update all the fields with all the information collected previously and save the file as `secrets.toml`

  Launch your streamlit application:
  ```
  streamlit run streamlit_vespa_app.py
  ```
  # Testing the Application

  You can try a mix of questions like:

  `What is the weather in Toronto ?`

  Followed by:

  `I'm looking for a screwdriver`

  And then:

  `Which one do you recommend to fix a watch?`
