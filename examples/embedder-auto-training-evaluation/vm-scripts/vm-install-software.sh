set -o pipefail

sudo yum-config-manager --enable \* || (echo "Enabling all repos failed" && exit 1)
sudo yum-config-manager --disable treasuredata-public || (echo "Disabling treasuredata repo failed" && exit 1)

sudo yum --assumeyes install git tmux || (echo "Installation failed" && exit 1)
tmux

wget https://github.com/vespa-engine/vespa/releases/download/v8.184.20/vespa-cli_8.184.20_linux_amd64.tar.gz -O vespa.tar.gz || (echo "Couldn't download Vespa CLI" && exit 1)
mkdir vespa && tar xf vespa.tar.gz -C vespa --strip-components 1 || (echo "Extracting tarball failed" && exit 1)
rm vespa.tar.gz 
mkdir bin
mv vespa/bin/vespa bin

echo "vespa config set target cloud"
echo "vespa config set application summer-project-2023.doc.default"
echo "vespa config get"
echo "vespa auth login"
echo "vespa auth cert application-package -f"
echo "vespa deploy application-package --wait 600"
