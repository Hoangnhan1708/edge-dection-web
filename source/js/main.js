var inputCate = document.getElementById("input__category-select")
categorySelected = "gradient_custom"
inputCate.onchange = function(event){
    categorySelected = event.target.value
}

function previewImage() {
   
    var file = document.getElementById('files').files[0];
    var reader = new FileReader();


    
    reader.onloadend = function () {
      sendToBackend(reader.result,categorySelected);
    };
  
    if (file) {
      reader.readAsDataURL(file);
    } else {
      preview.src = '';
    }
  }
  
  function sendToBackend(imageData, typeAlgorithm) {
    // Gửi dữ liệu hình ảnh imageData tới backend sử dụng fetch API
    fetch('http://127.0.0.1:5000/process_image', {
      method: 'POST',
      body: JSON.stringify({ image_data: imageData}),
      headers: {
        'Content-Type': 'application/json'
      }
    })
    .then(response => response.json())
    .then(data => {
      // Nhận hình ảnh từ server ở đây
      console.log('Kết quả xử lý từ backend:', data);
      var img = document.querySelector('.output__img');
      img.src = 'data:image/png;base64,' + data.processed_image;
      var h3 = document.querySelector('.output__heading');
      h3.innerText = "This is your image after edge detection"
      // Tìm phần tử có class là 'output-container' và thêm thẻ <img> vào đó
      var outputContainer = document.querySelector('.output-container');
      outputContainer.classList.add("show")
      outputContainer.appendChild(h3);
    })
    .catch(error => console.error('Lỗi:', error));
  }