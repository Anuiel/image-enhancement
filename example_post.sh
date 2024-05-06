curl -X 'POST' \
  'http://localhost:8092/enhance' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@lena.png' \
  -o 'lena228.png' \
  -v