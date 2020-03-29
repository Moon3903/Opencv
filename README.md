deteksi garis
Pertama yang dilakukan adalah mengambil gambar/frame. 
Setelah itu mensetting mask gambar. 
Ada GaussianBlur untuk mengurangi noise. 
lalu di threshold dengan thresh_binary. 
ada dilate dan erode untuk memperjelas gambar dan menghilangkan noise. 
setelah itu menggunakan HoughLinesP untuk mendapatkan garis pada gambar. 
terakhir adalah menampilkan hasil dari deteksi
