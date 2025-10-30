// Preview uploaded image
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("fileInput");
  const preview = document.getElementById("preview");

  input.addEventListener("change", () => {
    preview.innerHTML = "";
    const file = input.files[0];
    if (file) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.style.maxWidth = "300px";
      img.style.marginTop = "20px";
      preview.appendChild(img);
    }
  });
});
