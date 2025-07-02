### Adding a Lucene Analyzer component for Korean

To deploy, run:
```
mvn package
vespa deploy target/application
echo '{"id":"id:test:simple::korean-1","fields":{"title":"아름다운우리나라강산"}}' | vespa feed -
vespa query 'select * from simple where true' summary=dbg
vespa query 'select * from simple where title contains "아름다운우리나라강산"' language=ko summary=dbg
```
