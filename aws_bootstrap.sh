#!/bin/bash -e
## Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

export VESPA_HOME=/opt/vespa
configserver_addr=$1

if [ $# -eq 0 ]; then
        configserver_addr=localhost
fi

echo "Installing Vespa"

cat << 'EOF' > /etc/yum.repos.d/vespa-engine-stable.repo
[vespa-engine-stable]
name=vespa-engine-stable
baseurl=https://yahoo.bintray.com/vespa-engine/centos/$releasever/stable/$basearch
gpgcheck=0
repo_gpgcheck=0
enabled=1
EOF

yum -y install yum-utils epel-release
yum -y install vespa bind-utils git

fqdn=$(nslookup $(hostname) |grep Name |awk '{print $2}')
echo "Setting hostname to fqdn '$fqdn'"
hostnamectl set-hostname $fqdn

echo "Setting VESPA_CONFIGSERVERS=$configserver_addr"
echo "override VESPA_CONFIGSERVERS $configserver_addr" >> $VESPA_HOME/conf/vespa/default-env.txt

if [ "$configserver_addr" = "$fqdn" ]; then
	echo "Starting vespa configuration server "
	service vespa-configserver start
else
    echo "Starting vespa services"
    service vespa start
fi
