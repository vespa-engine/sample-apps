const aws = require("aws-sdk");
const s3 = new aws.S3({ apiVersion: "2006-03-01" });
const https = require("https");
const ZstdCodec = require("zstd-codec").ZstdCodec;
const TextDecoder = require("text-encoding").TextDecoder;

const vespaEndpoint =
  "vespacloud-docsearch.vespa-team.aws-us-east-1c.z.vespa-app.cloud";

const publicCert = `-----BEGIN CERTIFICATE-----
MIIEuDCCAqACCQCfUmlZ/6aw0DANBgkqhkiG9w0BAQsFADAeMRwwGgYDVQQDDBNj
bG91ZC52ZXNwYS5leGFtcGxlMB4XDTIwMTIyMjEwMTgwOFoXDTIxMTExNzEwMTgw
OFowHjEcMBoGA1UEAwwTY2xvdWQudmVzcGEuZXhhbXBsZTCCAiIwDQYJKoZIhvcN
AQEBBQADggIPADCCAgoCggIBALCFctSOlo1N442FlgikySsabEpX8854V1m1u02v
kMn8ZwijD1eOsPa7g1cs8hd7PN1SD+4MkIDxZZMovtQ1GxIwwCCkH1WM2RsE6rjs
i2lefMep35IeFORyq8ymn9S5GI4q/tWNwRu/n67sFvYzie1IzJuSre7azyL1uSC/
4pxVVU1ZqJIpMks3pRPITUIUpTUSfSkRuXgEPyYc8aJ8k1+1y07W7RvgnAkQlDNU
aBo97GlidQngp97vRn93628vLz0pbcgd9BzKKLIpr9TtGUxa9fh8gK112dgQR0fN
jm2uCbm2RfkLN+FNC4b3I+tQDdKLJDVv9tzrV1YvvC9+rb1+fkE3QIT5UF+gTeWT
RNknNkFws9lQjEaabIQvV7OvNUolpm3hYt5R8VqZMjZ2QUKMHiDFLIlcQ5T/25in
rZszJHMicxQshIHY9nyXErSmIK6VlSoTcSiLv/XeeyPMYCdqtUSu9/R7a7tVid4E
l1VvMPt/c1FWxRKG8Rmmnpa+7qo56VJRTZJHRWCq8bJ/fzxIIB3Ns4YnpoqFAjPP
75J+y0qwYd5cxSPUVfVtEYQcI0XV8GjSHmwq/FcUCtMGND/3r9MynALVpD6umoyV
HPJu5JYzvd1InLiWbA0RMoZKhLvkadddPfEka9/CwQV4UCvhYBp7F9tMPUsdCTSR
Z5ELAgMBAAEwDQYJKoZIhvcNAQELBQADggIBAImXTQfIw6MqyhU5+v4Dzfbl8fHU
U/DcDyen8U1nxLbQTPmyGqHVFwbdg6zfl6323jFTTuyDIGpyx07uX0Lyy8XtW9hG
L9QxkGs1FUFFFx0nSlSC5P0VjaB9YWOQ7/ZLyUXt5iD/n9FnWw/9kB69xrPvjcdQ
8YwvYwN0x2OiGDZf88RLvo+YMRDY7KvQI92Lilo0kP/5iOO2siAIe5ZkFEKfjADL
NtYcTD8Aa0nNnKE67CbXZP2ElpLNLeaU6fK4BiQGvULCJCSMrppjLh80jFRx4uDi
P49LpZ/xdDz8sJLeuN1aWjy3wwLIeze7cnJVfrkcWOjz/iM1nDH6ALPidKevP9Pq
5tnyv6O8a85jSM8UcBp/X1jk24sLY8WPOUjE1GpeNk7N7S5HJo+HuvcPZdZ1jm9/
UCkn6eGj+b+cOq2qDGqKbnKFS4DBWJzBV+a76n25UY61UQYhL4xvUxWFw4nmXCBI
TDTygNEUu58Y0fNsFr1tUSxAQg13y9pKZlsvLXwzmHmiiuYZrX9h2/m28Kb7DUq7
goB2RsikIOJxqHAuAgsqkol8oQyMBsApBx67smC80AWHvY93jJBzs4cSSNux9CI+
oIG8gdW9c+u6k8vqXfZ365Bv7kUlpIgGtfkzHIu0gbyoyKT7sPCrQfyiLHKOb7b6
bmRoTkkHVoUFYFt6
-----END CERTIFICATE-----`;

const listPrefixedKeys = ({ Bucket, Prefix, MaxKeys, RequestPayer }) => {
  console.log(
    `Listing keys with prefix ${Prefix} in bucket ${Bucket} with RequestPyer ${RequestPayer}. limited to ${MaxKeys} keys`
  );
  return s3
    .listObjectsV2({ Bucket, Prefix, MaxKeys, RequestPayer })
    .promise()
    .then(({ Contents }) =>
      Contents.map((content) => ({ Bucket, Key: content.Key, RequestPayer }))
    );
};

const getDateFilter = (date) => {
  const dateString = date
    .toISOString()
    .replace(/^(\d{4})-(\d{2})-(\d{2})T[^Z]+Z/, "$1$2$3");
  const re = new RegExp(
    `vespa-team\/vespacloud-docsearch\/default\/[^\/]+\/logs\/access\/JsonAccessLog\.default\.${dateString}\\d+\.zst`
  );

  return ({ Key }) => re.test(Key);
};

const getObjectData = ({ Bucket, Key, RequestPayer }) => {
  console.log(
    `Getting object with key ${Key} from bucket ${Bucket} with RequestPayer ${RequestPayer}`
  );
  return s3
    .getObject({ Bucket, Key, RequestPayer })
    .promise()
    .then((res) => ({ Bucket, Key, Body: res.Body }));
};

const decompress = ({ Bucket, Key, Body }) =>
  new Promise((resolve, reject) => {
    console.log(`Decompressing buffer`);
    ZstdCodec.run((zstd) => {
      try {
        const simple = new zstd.Streaming();
        const data = new TextDecoder().decode(simple.decompress(Body));
        resolve(data);
      } catch (err) {
        console.error(
          `Unable to decompress object with Key ${Key} in Bucket ${Bucket}`
        );
        reject(Error(err));
      }
    });
  });

const extractInput = (uri) =>
  decodeURIComponent(
    uri
      .match(/input=(.*)&jsoncallback/)[1]
      .replace(/%22/g, '"')
      .replace(/%5C/g, "\\")
      .replace(/%08|%09|%0A|%0B|%0C/g, "")
  );

const formatQueries = (logFile) =>
  logFile
    .split(/(?:\r\n|\r|\n)/g)
    .filter((line) => line.match(/input=(.*)&jsoncallback/))
    .map(JSON.parse)
    .map((obj) => ({ input: extractInput(obj.uri), time: obj.time }))
    .map((obj) => ({ fields: obj }));

const getSSMParameter = (parameter) => {
  console.log(`Getting parameter ${parameter}`);
  const ssm = new aws.SSM();
  return new Promise((resolve, reject) =>
    ssm.getParameter(
      {
        Name: parameter,
        WithDecryption: true,
      },
      (err, data) => (err ? reject(err) : resolve(data))
    )
  );
};

const feedQuery = (query, privateKey) => {
  //console.log("Feeding data to Vespa Cloud");
  const queryPath = "/document/v1/query/query/docid/";
  const data = JSON.stringify(query);

  return new Promise((resolve, reject) => {
    const options = {
      hostname: vespaEndpoint,
      port: 443,
      path: queryPath + query.fields.time,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": data.length,
      },
      key: privateKey,
      cert: publicCert,
    };

    const req = https.request(options, (res) =>
      res.on("data", (data) => {
        if (res.statusCode === 200) {
          console.log(data.toString());
          resolve();
        } else {
          reject(
            Error(`Status code: ${res.statusCode}, Message: ${data.toString()}`)
          );
        }
      })
    );

    req.on("error", reject);

    req.write(data);

    req.end();
  });
};

const feedQueries = (queries) =>
  getSSMParameter("VESPA_TEAM_DATA_PLANE_PRIVATE_KEY").then((parameter) =>
    Promise.all(
      queries.map((query) => feedQuery(query, parameter.Parameter.Value))
    )
  );

exports.handler = async (event, context) => {
  const Bucket = "vespa-cloud-data-prod.aws-us-east-1c-9eb633";
  const Prefix = "vespa-team/vespacloud-docsearch/default/";
  const MaxKeys = 15000;
  const RequestPayer = "requester";

  return listPrefixedKeys({ Bucket, Prefix, MaxKeys, RequestPayer })
    .then(
      (objects) =>
        objects.filter(getDateFilter(new Date(Date.now() - 86400000)))
        // objects.filter(getDateFilter(new Date(2021, 3, 6))) // For testing
    )
    .then((objects) => Promise.all(objects.map(getObjectData)))
    .then((objects) => Promise.allSettled(objects.map(decompress)))
    .then((promises) =>
      promises
        .filter((promise) => promise.status === "fulfilled")
        .map((promise) => promise.value)
    )
    .then((logFiles) => logFiles.map(formatQueries).flat())
    .then(feedQueries)
    .then((res) => {
      return { statusCode: 200 };
    })
    .catch((err) => ({ statusCode: 500, message: err.stack }));
};
