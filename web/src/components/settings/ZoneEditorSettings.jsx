function ZoneEditorSettings({
  phases,
  editingStation,
  setEditingStation,
  selectedZone,
  setSelectedZone,
  visibleZones,
  draftPoints,
  savingZones,
  onLoadStation,
  onUndo,
  onClear,
  onSave,
  onDelete,
  onAutoCalibrate,
  calibrating,
}) {
  return (
    <div className="tool-panel">
      <h3>Zone Editor</h3>
      <label>
        Edit Station
        <select value={editingStation} onChange={(e) => setEditingStation(e.target.value)}>
          {phases.map((station) => (
            <option key={station} value={station}>
              {station}
            </option>
          ))}
        </select>
      </label>
      <div className="button-row">
        <button onClick={onLoadStation} disabled={savingZones}>
          Load Station
        </button>
        <button onClick={onUndo} disabled={savingZones || draftPoints.length === 0}>
          Undo
        </button>
        <button onClick={onClear} disabled={savingZones || draftPoints.length === 0}>
          Clear
        </button>
      </div>
      <div className="button-row">
        <button onClick={onSave} disabled={savingZones}>
          {savingZones ? "Saving..." : "Save Station Polygon"}
        </button>
        <button onClick={onDelete} disabled={savingZones}>
          Delete Station Polygon
        </button>
      </div>
      <div className="button-row">
        <button onClick={onAutoCalibrate} disabled={savingZones || calibrating}>
          {calibrating ? "Calibrating..." : `Auto-Calibrate HSV for ${editingStation}`}
        </button>
      </div>
      <p className="muted">Draft points: {draftPoints.length}</p>
      <label>
        View Zone
        <select value={selectedZone} onChange={(e) => setSelectedZone(e.target.value)}>
          <option value="ALL">ALL</option>
          {phases.map((station) => (
            <option key={station} value={station}>
              {station}
            </option>
          ))}
        </select>
      </label>
      <div className="zone-list">
        {visibleZones.length === 0 ? (
          <p>No polygons loaded.</p>
        ) : (
          visibleZones.map((zone, idx) => (
            <div className="zone-chip" key={`${zone.station}-${idx}`}>
              {zone.station} ({zone.points_norm.length} points)
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default ZoneEditorSettings;
