var inputCate = document.getElementById("input__category-select")
var notify = document.querySelector("#notify")
var outputContainer = document.querySelector('.output-container');
var imgInputContainer = document.querySelector(".img-input-container")
var imgInput = document.querySelector('.img-input__img')
var headingImgInput = document.querySelector('.img-input__heading')
categorySelected = "gradient_custom"
inputCate.onchange = function(event){
    categorySelected = event.target.value
}

function previewImage() {
    
    notify.innerText = "Please wait..."
   
    var file = document.getElementById('files').files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        imgInputContainer.classList.add("show")
        imgInput.src = reader.result
        headingImgInput.innerText = "This is your old image"
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
      body: JSON.stringify({ image_data: imageData , type_algorithm :  typeAlgorithm}),
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
      
      outputContainer.classList.add("show")
      outputContainer.appendChild(h3);
      notify.innerText = ""
    })
    .catch(error => console.error('Lỗi:', error));
  }