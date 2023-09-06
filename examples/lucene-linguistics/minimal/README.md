# Minimal `lucene-linguistics` setup

This application package contains a bare minimal setup to get started with the `lucene-linguistics`.

NOTE: the `ext/linguistics` directory can't be empty.
The empty `ext/linguistics/dummy.txt` file serves no purpose except to make the Vespa configuration system happy ([details](https://github.com/vespa-engine/vespa/issues/27912)).
When any other files are added to that directory then you could remove the `dummy.txt`.
