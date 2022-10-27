#### This is the last step that should happen after finishing CI/CD
docker stop hnx || true
docker rm hnx || true
rm ghtoken.txt
