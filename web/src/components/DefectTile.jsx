function DefectTile({ frameTick, onOpenDefectSettingsModal, defectEnabled, currentPhase }) {
  return (
    <article className="tile">
      <div className="tile-head">
        <h2>Defect Camera + Reference</h2>
      </div>

      <div className="pill-row">
        <button className="pill-btn" onClick={onOpenDefectSettingsModal}>
          Defect Camera Settings
        </button>
      </div>

      <div className="dual-feed">
        <div>
          <h3>Defect Cam</h3>
          {defectEnabled ? (
            <img
              className="feed-image"
              src={`/api/vision/defect/frame?t=${frameTick}`}
              alt="Defect camera feed"
            />
          ) : (
            <div className="feed-image feed-paused">
              <div>Defect camera idle.</div>
              <div>Auto-enables at phase: Test</div>
              <div>Current phase: {currentPhase}</div>
            </div>
          )}
        </div>
        <div>
          <h3>Reference Image</h3>
          <img
            className="feed-image"
            src={`/api/vision/defect/reference-image?t=${frameTick}`}
            alt="QC reference"
          />
        </div>
      </div>
    </article>
  );
}

export default DefectTile;
