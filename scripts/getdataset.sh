#Run as sudo
mkdir ~/DataSets
cd ~/DataSets
sudo yum install p7zip p7zip-pluginsi -y
curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash
unzip *.zip
