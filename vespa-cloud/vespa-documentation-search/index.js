const aws = require("aws-sdk");
const s3 = new aws.S3({ apiVersion: "2006-03-01" });
const https = require("https");
const ZstdCodec = require("zstd-codec").ZstdCodec;
const TextDecoder = require("text-encoding").TextDecoder;

const vespaHostname =
  "my-instance.search-suggestion.chunnoo.aws-us-east-1c.dev.z.vespa-app.cloud";

const publicCert = `-----BEGIN CERTIFICATE-----
MIIEuDCCAqACCQDuqW9lydTQNDANBgkqhkiG9w0BAQsFADAeMRwwGgYDVQQDDBNj
bG91ZC52ZXNwYS5leGFtcGxlMB4XDTIxMDYyNDA3MDY0NFoXDTIxMDcwODA3MDY0
NFowHjEcMBoGA1UEAwwTY2xvdWQudmVzcGEuZXhhbXBsZTCCAiIwDQYJKoZIhvcN
AQEBBQADggIPADCCAgoCggIBAOi27q1XwY+s+caS2tTcPqIscOY1DQmJmqwMY0U8
W4JFNllQkgMZr8xELveM3B+UfKvaBOGrtIL6MRUVWzbG6HtzRNGJkQbQGFtD5utx
5HY8GD5YrYzOmir5E+RUYoHsqkZNs9dUAaaajF3KMOTNx0DjRwdxxgrRAJPdMi+7
lMgAkm+WZG1IGqDjN344M4YdZzS08lR92pyLIInJZpPIVaJphW5Wu9cRN5R0q5uo
BlbobE+RF+6SBXuQRUctmVUgiOA98EfsXvQiiYF6GLuQTP3dPLl7RiQ0VHUNpVJ+
NU62nE0/Z9VP+ZDmsq/LPwBT93h2iRPPxNu/5dy1hJttdA3SjVe16xGuJA3S4FeG
2PxDfGOiYTpIgaA7k8coXSLcSDSY19LUqkTHMG80y/Pf2gfVpX/wGKuJ3Mpc/seY
Odkcg+VO06yWhJE9g7OXCQx/BXc2+RF0cwHdXEHknLrderfS/h6okMOiIYIum+fP
hCPZuobQ6Sa93wjema6Qsf0Y5fiZMc0bwNjMVJTgfLjrRqc7qDU91dyn4ot+IaUR
3wOkSU4m56YiHJYFCraozrDlU43G+hlW6hxeoypZGXPlxVyUri8EU4FRR78DEG8A
N56fLx4GHr2pfzelKcH+FlrwhW5lDeHnPiYvZV+AwPZOUCx6yIXaeBKQdIqACcyq
p2XNAgMBAAEwDQYJKoZIhvcNAQELBQADggIBAL1/wwxJPJrl21Af4SDTcQYeY5aE
FulYb2av8QYEzGF9jk62669SwXgSMuoMXr+ZMU2YqSkYH/9jz2pFFiUzcysuR1VU
4+HqNj+T5P6TgzeAOyYILZMI7ho7PkzyexOVzE5QzYwGR/gh9/uCkLV++tr0r7kC
vij6lTgixZDbExmj/W9mFDswoXKnm22FceES/T213wYeZUwaDek1exRHh/UUaSgh
g5QMPX0sw4gtgx/x3duQbzFygG7Xd7JCFHnY48ZgQ4UQHzGJuBCA/1KOHVJ3TrqY
P9ln5Rb082tXKtN8odStS290ODB4WfXSPPLsMYL6BHd2oMzgg/LpHjWeHUGK0gr5
Q1ccNxXCcCtl3GJk34ntkZy+10VKzup1awh1x/JltQeo1UdM0HoQLEc01fyJE6TK
ZTcXRERWFVoujKZB1D/NPkL6E3Mxj6yPFFIWTCr6BeHgHEyeU+geotIHofUCbMvT
xdRGQznua7WCPno4nVDiUQkXCUU0GO/AI9DvT/9UrQCwyg7n56JUM4taSWyyGWLk
SeD9CFJni7e7sYCfxU3F8JVcpSEY8JGWifrXKBkn1gdDh9VydUZ8bPKJR37nUmKa
XG68ibZCW9B6m9rQlMcMaVUSNTGH6qpJ0KI3G63FX7d6qSRIZNpI2G8oWZCHidBM
KvReil2oZAclH+d0
-----END CERTIFICATE-----`;

const getObjectData = ({ Bucket, Key }) => {
  console.log(`Getting object with key ${Key} from bucket ${Bucket}`);
  return s3
    .getObject({ Bucket, Key })
    .promise()
    .then((res) => res.Body);
};

const decompress = (buffer) =>
  new Promise((resolve, reject) => {
    ZstdCodec.run((zstd) => {
      try {
        const simple = new zstd.Simple();
        const data = new TextDecoder().decode(simple.decompress(buffer));
        resolve(data);
      } catch (err) {
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
  console.log("Feeding query to Vespa");
  const queryPath = "/document/v1/query/query/group/0/";
  const data = JSON.stringify(query);

  return new Promise((resolve, reject) => {
    const options = {
      hostname: vespaHostname,
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
  getSSMParameter("AccessLogPrivateKey").then((parameter) =>
    Promise.all(
      queries.map((query) => feedQuery(query, parameter.Parameter.Value))
    )
  );

exports.handler = async (event, context) => {
  const options = {
    Bucket: event.Records[0].s3.bucket.name,
    Key: decodeURIComponent(event.Records[0].s3.object.key.replace(/\+/g, " ")),
  };

  return getObjectData(options)
    .then(decompress)
    .then((logFile) => feedQueries(formatQueries(logFile)))
    .then((res) => {
      return { statusCode: 200 };
    })
    .catch((err) => ({ statusCode: 500, message: err.stack }));
};
