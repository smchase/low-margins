export async function fetchMNISTImages(url: string) {
  const res = await fetch(url);
  const buf = await res.arrayBuffer();
  const view = new DataView(buf);

  const magic = view.getUint32(0, false);
  if (magic !== 0x00000803) throw new Error("bad images magic");

  const numImages = view.getUint32(4, false);
  const numRows = view.getUint32(8, false);
  const numCols = view.getUint32(12, false);

  const images: Uint8Array[] = [];
  const imageSize = numRows * numCols;
  let offset = 16;

  for (let i = 0; i < numImages; i++) {
    const img = new Uint8Array(buf, offset, imageSize);
    // copy if you want to detach:
    images.push(new Uint8Array(img));
    offset += imageSize;
  }

  return { images, numImages, numRows, numCols };
}

export async function fetchMNISTLabels(url: string) {
  const res = await fetch(url);
  const buf = await res.arrayBuffer();
  const view = new DataView(buf);

  const magic = view.getUint32(0, false);
  if (magic !== 0x00000801) throw new Error("bad labels magic");

  const numLabels = view.getUint32(4, false);
  const labels = new Uint8Array(buf, 8, numLabels);
  return { labels, numLabels };
}
