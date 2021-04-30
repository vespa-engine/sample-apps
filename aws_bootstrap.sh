#!/bin/bash -e
## Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

export VESPA_HOME=/opt/vespa
configserver_addr=$1

if [ $# -eq 0 ]; then
        configserver_addr=localhost
fi

echo "Installing Vespa"
yum -y install yum-utils epel-release
yum-config-manager --add-repo https://copr.fedorainfracloud.org/coprs/g/vespa/vespa/repo/epel-7/group_vespa-vespa-epel-7.repo
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
