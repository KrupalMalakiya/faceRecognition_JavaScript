const imageUpload = document.getElementById("imageUpload");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(start);

async function start() {
  const container = document.createElement("div");
  container.style.position = "relative";
  // container.style.width = "40px";
  // container.style.height = "40px";
  document.body.append(container);
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.5);
  let image;
  let canvas;
  document.body.append("Loaded");
  imageUpload.addEventListener("change", async () => {
    if (image) image.remove();
    if (canvas) canvas.remove();
    image = await faceapi.bufferToImage(imageUpload.files[0]);
    container.append(image);
    canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
      });
      console.log(
        "------------------------IMAGE NAME:----------",
        result.toString()
      );
      drawBox.draw(canvas);
    });
  });
}

function loadLabeledImages() {
  const labels = [
    "Aditya",
    // "Anila",
    "Archan",
    "Gabriel_bheekar",
    "Krupal",
    "Ritesh",
    "Satish",
    "Yash Fullstack",
    "Yash hingu",
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(`./images/${label}/${i}.jpg`);

        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        descriptions.push(detections?.descriptor);

        console.log("Image :", label, " No: ", i, "IMAGE DATA");
        console.log("Detections:", detections);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
