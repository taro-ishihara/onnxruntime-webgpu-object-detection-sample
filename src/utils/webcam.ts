class Webcam {
  private static _stream: MediaStream

  static get stream() {
    return this._stream
  }

  static get imageCapture() {
    const track = this._stream.getVideoTracks()[0]
    return new ImageCapture(track)
  }

  private constructor() {}

  public static async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({
      video: true,
    })
    return this._stream
  }

  public static stop() {
    if (this._stream) {
      this._stream.getVideoTracks().forEach((track) => {
        track.stop()
      })
    }
  }
}

export { Webcam }
