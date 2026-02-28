function pointsToSvg(pointsNorm, frameWidth, frameHeight) {
  return pointsNorm
    .map(([x, y]) => `${Math.round(x * frameWidth)},${Math.round(y * frameHeight)}`)
    .join(" ");
}

function LiveFeedTile({
  currentPhase,
  frameTick,
  liveStatus,
  frameWidth,
  frameHeight,
  visibleZones,
  draftPoints,
  onLiveFrameClick,
  onOpenZoneModal,
  onOpenLiveSettingsModal,
}) {
  return (
    <article className="tile">
      <div className="tile-head">
        <h2>Live Feed: Line Camera</h2>
        <span className="badge">{currentPhase}</span>
      </div>

      <div className="pill-row">
        <button className="pill-btn" onClick={onOpenZoneModal}>
          Zone Settings
        </button>
        <button className="pill-btn" onClick={onOpenLiveSettingsModal}>
          HSV Settings
        </button>
      </div>

      <div className="feed-frame" onClick={onLiveFrameClick} role="button" tabIndex={0}>
        <img
          className="feed-image"
          src={`/api/vision/live/frame?t=${frameTick}`}
          alt="Live line feed"
        />
        <svg
          className="zone-overlay"
          viewBox={`0 0 ${frameWidth} ${frameHeight}`}
          preserveAspectRatio="none"
        >
          {visibleZones.map((zone, idx) => (
            <g key={`${zone.station}-${idx}`}>
              <polygon
                className="zone-polygon"
                points={pointsToSvg(zone.points_norm, frameWidth, frameHeight)}
              />
            </g>
          ))}
          {draftPoints.length > 0 ? (
            <>
              <polyline
                className="zone-draft"
                points={pointsToSvg(draftPoints, frameWidth, frameHeight)}
              />
              {draftPoints.map(([x, y], idx) => (
                <circle
                  key={`draft-${idx}`}
                  cx={Math.round(x * frameWidth)}
                  cy={Math.round(y * frameHeight)}
                  r="4"
                  className="zone-point"
                />
              ))}
            </>
          ) : null}
        </svg>
      </div>
      <p className="click-hint">Click on live feed to add polygon points for selected station.</p>

      {liveStatus ? (
        <div className="status-row">
          <span>Dominant: {liveStatus.dominant_station}</span>
          <span>Score: {Number(liveStatus.dominant_score || 0).toFixed(2)}</span>
          <span>
            Candidates:{" "}
            {liveStatus.instant_candidates?.length
              ? liveStatus.instant_candidates.join(", ")
              : "NOT_PRESENT"}
          </span>
        </div>
      ) : null}
    </article>
  );
}

export default LiveFeedTile;
