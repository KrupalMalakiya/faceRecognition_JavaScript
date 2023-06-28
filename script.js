const imageUpload = document.getElementById("imageUpload");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("./models"),
])
  .then(start)
  .catch((err) => {
    console.log(err);
  });

async function start() {
  const container = document.createElement("div");
  const labeledDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
  container.style.position = "relative";
  document.body.append(container);
  document.body.append("Loaded");

  imageUpload.addEventListener("change", async () => {
    const image = await faceapi.bufferToImage(imageUpload.files[0]);
    container.append(image);
    const canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
    const displaySize = {
      width: image.width,
      height: image.height,
    };
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetection = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetection.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    results.forEach((result, i) => {
      const box = resizedDetection[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
      });
      drawBox.draw(canvas);
    });

    document.body.append(detections.length);
  });
}

function loadLabeledImages() {
  const lables = [
    "Aditya",
    "Anil",
    // "Archan",
    // "Gabriel_bheekar",
    // "Krupal",
    // "Ritesh",
    // "Satish",
    // "Yash Fullstack",
    // "Yash hingu",
  ];

  return Promise.all(
    lables.map(async (lable) => {
      const discriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `https://github.com/KrupalMalakiya/faceRecognition_JavaScript/tree/master/images/${lable}/${i}.jpg`
        );

        const detection = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptors();

        discriptions.push(detection.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(lable, discriptions);
    })
  );
}
