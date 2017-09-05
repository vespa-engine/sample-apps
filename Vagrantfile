# Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
Vagrant.configure("2") do |config|

  config.vm.box = "boxcutter/centos73-desktop"

  config.ssh.forward_agent = true

  config.vm.synced_folder ".", "/sample-apps"

  config.vm.provider "virtualbox" do |vb|
    vb.gui = true
    vb.name = "vespa-sample-apps"

    vb.memory = "3074"
    vb.cpus = 1
  end

  config.vm.provision "shell", env: {"VESPA_HOME" => "/opt/vespa", "PATH" => "$PATH:/opt/vespa/bin"}, inline: <<-SHELL
    # Install latest Vespa release
    yum -y install yum-utils epel-release
    yum-config-manager --add-repo https://copr.fedorainfracloud.org/coprs/g/vespa/vespa/repo/epel-7/group_vespa-vespa-epel-7.repo
    yum -y install vespa

    # Vespa requires hostname being fully qualified
    hostnamectl set-hostname `hostname -f`

    # Start Vespa config server
    echo "override VESPA_CONFIGSERVERS `hostname`" >> $VESPA_HOME/conf/vespa/default-env.txt
    service vespa-configserver start
  SHELL
end
