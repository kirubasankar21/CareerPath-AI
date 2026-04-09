/**
 * Drag-and-drop + label for resume file input on dashboard.
 */
(function () {
  var form = document.getElementById("resume-upload-form");
  if (!form) return;

  var zone = document.getElementById("resume-dropzone");
  var input = document.getElementById("resume-file");
  var label = document.getElementById("resume-file-label");
  if (!zone || !input) return;

  function showName(file) {
    if (!label) return;
    if (file && file.name) {
      label.textContent = "Selected: " + file.name;
      label.hidden = false;
    } else {
      label.textContent = "";
      label.hidden = true;
    }
  }

  input.addEventListener("change", function () {
    showName(input.files && input.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (ev) {
    zone.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add("resume-dropzone--active");
    });
  });

  ["dragleave", "drop"].forEach(function (ev) {
    zone.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove("resume-dropzone--active");
    });
  });

  zone.addEventListener("drop", function (e) {
    var files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    var file = files[0];
    var ok =
      file.type === "application/pdf" ||
      file.type === "text/plain" ||
      /\.pdf$/i.test(file.name) ||
      /\.txt$/i.test(file.name);
    if (!ok) {
      showName(null);
      alert("Please drop a PDF or TXT file.");
      return;
    }
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      showName(file);
    } catch (err) {
      showName(null);
    }
  });
})();
