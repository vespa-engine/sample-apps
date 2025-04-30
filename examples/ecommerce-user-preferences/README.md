<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Car Preferences Web Application

A web application that helps users find cars based on their preferences. The application uses GPT-4 to extract car preferences from natural language and Vespa for search rankings. Check out the [system prompt](webapp/system_prompt.txt) for details - and feel free to modify it!

## Features

- Interactive chat interface for expressing car preferences
- Real-time preferences extraction and visual representation
- Manual preference adjustment with sliders (-5 to +5 range)
- Car search results with detailed information
- Session-based conversation history

## Installation

To get the application running, you'll need to:

1. Set up the Vespa application under [app](app)
2. Feed the data into Vespa using Logstash and the provided [logstash.conf](logstash.conf)
3. Set up and run the web application under [webapp](webapp)

### Set up the Vespa application

1. Log in to [Vespa Cloud](https://cloud.vespa.ai) and create a tenant, if you don't have one already.

2. Deploy your application using the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):

```bash
# point Vespa CLI to Vespa Cloud
vespa config set target cloud

# also point it to your tenant and application
# If you don't have an application, it will be created on deploy
vespa config set application YOUR_TENANT_NAME.YOUR_APPLICATION_NAME

# authenticate Vespa CLI with your Vespa Cloud credentials
vespa auth cert

# go to the application package directory
# and set up the mTLS certificates
cd app
vespa auth cert

# deploy the application
vespa deploy --wait 900
```

**NOTE**: If you're running Vespa locally, you'd skip the security steps:

```bash
vespa config set target local
cd app
vespa deploy --wait 900
```

### Feed the data into Vespa

1. [Download Logstash](https://www.elastic.co/downloads/logstash), for example the `tar.gz` and unpack it.

2. Update [logstash.conf](logstash.conf):
   - **Use your Vespa endpoint**. You'll see it in the Vespa Cloud UI, or comment it out if you run a local Vespa
   - **Update the certificate files** to files from your local `.vespa` directory
   - **Point Logstash to the sample data** in [dataset](dataset/)

3. Run Logstash:

```bash
/PATH/TO/LOGSTASH/bin/logstash -f /PATH/TO/THIS/REPO/examples/ecommerce-user-preferences/logstash.conf
```

### Set up and run the web application

In the `webapp` directory:

1. Set up a conda environment (recommended):
```bash
conda env create -f environment.yml
conda activate car_preferences
```

2. Create a `.env` file in `webapp` with your OpenAI API key, Vespa application endpoint, and certificate files:
```
OPENAI_API_KEY=your_api_key_here
VESPA_API_URL=https://YOUR_VESPA_ENDPOINT
VESPA_CERT_PATH=/path/to/vespa/certificate
VESPA_KEY_PATH=/path/to/vespa/key
```

*NOTE*: If you're running Vespa locally, remove all references to `CERT_PATH` and `KEY_PATH` in [app.py](webapp/app.py).

3. Start the Flask server:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Type your car preferences in the chat input, for example:
   - "I want a cheap car that's fuel efficient"
   - "I'm looking for a manual transmission car, ideally an Audi A4"
   - "I need a family car with low mileage"

2. The system will analyze your preferences and:
   - Extract key preferences and assign importance weights
   - Show these preferences as colored pills above the search results
   - Display matching cars based on your preferences

3. Manual preference adjustment:
   - Click on any preference pill to open an adjustment slider
   - Use the slider to set your preferred weight from -5 to +5
   - Click "Apply" to update your search results with the new weight
   - Manual adjustments will be preserved in future chat interactions

4. Continue the conversation to refine your preferences and get better results.

5. Click "New Chat" to start over with a fresh conversation.

*NOTE*: The sample dataset doesn't contain images of cars, so the web application will display each specific model by fetching a Wikipedia image. There's some rudimentary caching, making some images load faster than others. This also means that all model years of the same model will display the same image (the first one we find in Wikipedia).

## Web Application Technical Details

- Frontend: HTML, CSS, JavaScript with Bootstrap 5
- Backend: Flask (Python)
- AI: OpenAI GPT-4o for preference extraction
- Search: Vespa for preference-based car ranking
- Authentication: mTLS for secure API communication (assuming you're using Vespa Cloud)
