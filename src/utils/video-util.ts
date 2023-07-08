const onloadmetadataPromise = (
  videoElement: HTMLVideoElement,
): Promise<void> => {
  return new Promise((resolve, reject) => {
    videoElement.onloadedmetadata = () => {
      resolve()
    }
    videoElement.onerror = (error) => {
      reject(error)
    }
  })
}

export { onloadmetadataPromise }
