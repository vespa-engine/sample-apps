<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - search as you type

* `git clone git@github.com:vespa-engine/sample-apps.git`
* `git clone git@github.com:vespa-engine/documentation.git`
* `cd documentation/`
* `bundle exec jekyll build -p _plugins-vespafeed`
* `cd ../sample-apps/incremental-search/search-as-you-type/`
* `mv ../../../documentation/open_index.json ./`
* `docker pull vespaengine/vespa`
* `docker run -m 6G --detach --name vespa --hostname vespa-example --publish 8080:8080 --publish 19071:19071 vespaengine/vespa`
* `curl -s --head http://localhost:19071/ApplicationStatus`
* `mvn clean package`
* `curl --header Content-Type:application/zip --data-binary @target/application.zip localhost:19071/application/v2/tenant/default/prepareandactivate`
* `curl -s --head http://localhost:8080/ApplicationStatus`
* `python3 feed_to_vespa.py`
* Open <http://localhost:8080/site/> in a browser.
