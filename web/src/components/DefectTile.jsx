function DefectTile({
  frameTick,
  onOpenDefectSettingsModal,
  defectEnabled,
  currentPhase,
  defectStatus,
  onNotifyOperator,
  notifyingOperator,
}) {
  const snapshotLocked = Boolean(defectStatus?.snapshot_locked);
  const qcResult = defectStatus?.qc_result;
  const qcLabel = !defectEnabled
    ? "IDLE"
    : qcResult?.status
      ? qcResult.status
      : snapshotLocked
        ? "LOCKED"
        : "SEARCHING";
  const qcClass = qcLabel === "PASS" ? "pass" : qcLabel === "FAIL" ? "fail" : "neutral";

  return (
    <article className="tile">
      <div className="tile-head">
        <h2>Defect Camera + Reference</h2>
        <div className="tile-head-actions">
          <button
            className="icon-btn"
            onClick={onOpenDefectSettingsModal}
            title="Defect Settings"
            aria-label="Defect Settings"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <path
                d="M10.6 2h2.8l.5 2.2c.6.2 1.2.4 1.8.7l1.9-1.2 2 2-1.2 1.9c.3.6.5 1.2.7 1.8L22 10.6v2.8l-2.2.5c-.2.6-.4 1.2-.7 1.8l1.2 1.9-2 2-1.9-1.2c-.6.3-1.2.5-1.8.7l-.5 2.2h-2.8l-.5-2.2c-.6-.2-1.2-.4-1.8-.7l-1.9 1.2-2-2 1.2-1.9c-.3-.6-.5-1.2-.7-1.8L2 13.4v-2.8l2.2-.5c.2-.6.4-1.2.7-1.8L3.7 6.4l2-2 1.9 1.2c.6-.3 1.2-.5 1.8-.7L10.6 2zm1.4 6a4 4 0 100 8 4 4 0 000-8z"
                fill="currentColor"
              />
            </svg>
          </button>
        </div>
      </div>

      <div className={`qc-pill ${qcClass}`}>QC: {qcLabel}</div>
      {defectEnabled && qcResult?.status === "FAIL" ? (
        <div className="button-row" style={{ marginTop: 4, marginBottom: 8 }}>
          <button onClick={onNotifyOperator} disabled={notifyingOperator}>
            {notifyingOperator ? "Notifying..." : "Notify Operator"}
          </button>
        </div>
      ) : null}
      {defectEnabled ? (
        <p className="muted">
          {snapshotLocked
            ? "Snapshot + QC frozen. Obstruct/idle camera to re-arm a fresh check."
            : "Searching for ArUco markers to lock snapshot."}
        </p>
      ) : null}

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
