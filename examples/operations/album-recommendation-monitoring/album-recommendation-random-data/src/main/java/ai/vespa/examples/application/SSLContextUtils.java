// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.application;

import org.apache.hc.core5.ssl.SSLContextBuilder;
import org.bouncycastle.asn1.pkcs.PKCSObjectIdentifiers;
import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.asn1.x509.AlgorithmIdentifier;
import org.bouncycastle.asn1.x9.X9ObjectIdentifiers;
import org.bouncycastle.cert.X509CertificateHolder;
import org.bouncycastle.cert.jcajce.JcaX509CertificateConverter;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.bouncycastle.openssl.PEMKeyPair;
import org.bouncycastle.openssl.PEMParser;

import javax.net.ssl.SSLContext;
import java.io.IOException;
import java.io.StringReader;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;
import java.security.KeyFactory;
import java.security.KeyStore;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.ArrayList;
import java.util.List;

/**
 * @author freva
 */
public class SSLContextUtils {
    private static final BouncyCastleProvider bcProvider = new BouncyCastleProvider();

    public static SSLContext sslContext(Path privateKeyPath, Path certificatePath) {
        try {
            PrivateKey privateKey = fromPemEncodedPrivateKey(Files.readString(privateKeyPath));
            X509Certificate certificate = fromPemEncodedCertificate(Files.readString(certificatePath));

            char[] password = new char[0];
            KeyStore keyStore = KeyStore.getInstance("PKCS12");
            keyStore.setKeyEntry("key", privateKey, password, new Certificate[]{ certificate });

            return new SSLContextBuilder()
                    .loadKeyMaterial(keyStore, password)
                    .build();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static PrivateKey fromPemEncodedPrivateKey(String pem) {
        try (PEMParser parser = new PEMParser(new StringReader(pem))) {
            List<Object> unknownObjects = new ArrayList<>();
            Object pemObject;
            while ((pemObject = parser.readObject()) != null) {
                if (pemObject instanceof PrivateKeyInfo) {
                    PrivateKeyInfo keyInfo = (PrivateKeyInfo) pemObject;
                    PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(keyInfo.getEncoded());
                    return createKeyFactory(keyInfo.getPrivateKeyAlgorithm())
                            .generatePrivate(keySpec);
                } else if (pemObject instanceof PEMKeyPair) {
                    PEMKeyPair pemKeypair = (PEMKeyPair) pemObject;
                    PrivateKeyInfo keyInfo = pemKeypair.getPrivateKeyInfo();
                    return createKeyFactory(keyInfo.getPrivateKeyAlgorithm())
                            .generatePrivate(new PKCS8EncodedKeySpec(keyInfo.getEncoded()));
                } else {
                    unknownObjects.add(pemObject);
                }
            }
            throw new IllegalArgumentException("Expected a private key, but found " + unknownObjects);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (GeneralSecurityException e) {
            throw new RuntimeException(e);
        }
    }

    private static KeyFactory createKeyFactory(AlgorithmIdentifier algorithm) throws NoSuchAlgorithmException {
        if (X9ObjectIdentifiers.id_ecPublicKey.equals(algorithm.getAlgorithm())) {
            return createKeyFactory("EC");
        } else if (PKCSObjectIdentifiers.rsaEncryption.equals(algorithm.getAlgorithm())) {
            return createKeyFactory("RSA");
        } else {
            throw new IllegalArgumentException("Unknown key algorithm: " + algorithm);
        }
    }

    private static KeyFactory createKeyFactory(String algorithm) throws NoSuchAlgorithmException {
        return KeyFactory.getInstance(algorithm, bcProvider);
    }

    public static X509Certificate fromPemEncodedCertificate(String pem) {
        try (PEMParser parser = new PEMParser(new StringReader(pem))) {
            return toX509Certificate(parser.readObject());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (CertificateException e) {
            throw new RuntimeException(e);
        }
    }

    private static X509Certificate toX509Certificate(Object pemObject) throws CertificateException {
        if (pemObject instanceof X509Certificate) return (X509Certificate) pemObject;
        if (pemObject instanceof X509CertificateHolder) {
            return new JcaX509CertificateConverter()
                    .setProvider(bcProvider)
                    .getCertificate((X509CertificateHolder) pemObject);
        }
        throw new IllegalArgumentException("Invalid type of PEM object: " + pemObject);
    }
}
