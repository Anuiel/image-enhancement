curl -X 'POST' \
  'http://localhost:8095/enhance' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@mop.jpg' \
  -o 'lena228.png' \
  -v