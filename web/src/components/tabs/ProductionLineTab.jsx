import LiveViewInlineSection from '../../live/LiveViewInlineSection'
import { getPriorityLabel, formatDate, OPERATION_SEQUENCE, PHASE_TO_OPERATION } from './tabHelpers'

export default function ProductionLineTab({
  schedule,
  productionStates,
  lineVisionPhase,
  onPhaseChange,
  onLoadOrder,
  onCompleteOperation,
  onResetFlow,
}) {
  const currentOperation = PHASE_TO_OPERATION[lineVisionPhase]

  const getLoadedState = (states) =>
    states.find((state) =>
      OPERATION_SEQUENCE.some((op) => state[op]?.required) &&
      !OPERATION_SEQUENCE.every((op) => !state[op]?.required || state[op]?.finished_at)
    )

  const loadedState = getLoadedState(productionStates)
  const loadedPlan = loadedState && schedule?.production_plans?.find(p => p.order_number === loadedState.production_order)

  const ops = [
    { key: 'smt', label: 'SMT', desc: 'Surface Mount Technology' },
    { key: 'reflow', label: 'Reflow', desc: 'Reflow Soldering' },
    { key: 'tht', label: 'THT', desc: 'Through-Hole Technology' },
    { key: 'aoi', label: 'AOI', desc: 'Automated Optical Inspection' },
    { key: 'test', label: 'Test', desc: 'Functional Testing' },
    { key: 'coating', label: 'Coating', desc: 'Conformal Coating' },
    { key: 'pack', label: 'Pack', desc: 'Final Packaging' },
  ]

  const phaseFlow = [
    { op: 'smt', label: 'SMT' },
    { op: 'reflow', label: 'Reflow' },
    { op: 'tht', label: 'THT' },
    { op: 'aoi', label: 'AOI' },
    { op: 'test', label: 'Test' },
    { op: 'coating', label: 'Coating' },
    { op: 'pack', label: 'Pack' },
  ]

  return (
    <div className="flex items-stretch gap-4" style={{ minHeight: 'calc(100vh - 170px)' }}>
      {/* ── Sidebar: Upcoming Orders ── */}
      <div className="w-80 flex-shrink-0">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden sticky top-6 h-[calc(100vh-170px)] flex flex-col">
          <div className="px-5 py-4 border-b border-gray-100 bg-gray-50">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">Upcoming Orders</h2>
              <span className="text-xs font-medium text-gray-500 bg-gray-200 rounded-full px-2 py-0.5">
                {schedule?.production_plans?.length || 0}
              </span>
            </div>
          </div>
          <div className="divide-y divide-gray-100 flex-1 min-h-0 overflow-y-auto">
            {schedule?.production_plans?.map((plan, index) => {
              const state = productionStates.find(s => s.production_order === plan.order_number)
              const isLoaded = state && OPERATION_SEQUENCE.some(op => state[op]?.required)
                && !OPERATION_SEQUENCE.every(op => !state[op]?.required || state[op]?.finished_at)
              const deadline = new Date(plan.deadline)
              const isLate = new Date(plan.ends_at) > deadline

              return (
                <div key={plan.order_number} className={`px-5 py-3 hover:bg-gray-50 transition cursor-default ${isLoaded ? 'bg-blue-50 border-l-4 border-l-blue-500' : ''}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-400 font-mono">#{index + 1}</span>
                      <span className="font-semibold text-sm text-gray-900">{plan.order_number}</span>
                    </div>
                    {isLoaded && (
                      <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" title="On Line"></span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mb-1.5">{plan.customer} · {plan.quantity}× {plan.product}</div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <span className={`inline-block w-1.5 h-1.5 rounded-full ${plan.priority <= 2 ? 'bg-red-400' : 'bg-green-400'}`}></span>
                      <span className="text-xs text-gray-400">P{plan.priority}</span>
                      <span className={`text-xs ${isLate ? 'text-red-500 font-medium' : 'text-gray-400'}`}>
                        · {deadline.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </span>
                    </div>
                    {!isLoaded && (
                      <button
                        onClick={() => onLoadOrder(plan.order_number)}
                        className="text-xs font-medium text-blue-600 hover:text-blue-800 transition"
                      >
                        Load →
                      </button>
                    )}
                  </div>
                </div>
              )
            })}
            {(!schedule?.production_plans || schedule.production_plans.length === 0) && (
              <div className="px-5 py-8 text-center text-sm text-gray-400">No orders scheduled</div>
            )}
          </div>
        </div>
      </div>

      {/* ── Main Area: Loaded Order ── */}
      <div className="flex-1 min-w-0">
        {!loadedState ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center py-20">
              <div className="mx-auto w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <svg className="h-8 w-8 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-700 mb-1">No Order on Line</h3>
              <p className="text-sm text-gray-400">Select an order from the sidebar to load it onto the production line</p>
            </div>
          </div>
        ) : (() => {
          const completedOps = ops.filter(o => loadedState[o.key]?.required && loadedState[o.key]?.finished_at).length
          const totalRequired = ops.filter(o => loadedState[o.key]?.required).length
          const progressPct = totalRequired > 0 ? Math.round((completedOps / totalRequired) * 100) : 0

          return (
            <div className="space-y-4">
              {/* Order Header Card */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                <div className="bg-gradient-to-r from-slate-800 to-slate-700 px-6 py-5 text-white">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <span className="text-xs font-medium uppercase tracking-wider text-slate-300">Active Order</span>
                        <span className="inline-flex items-center gap-1 text-xs font-medium bg-green-500/20 text-green-300 rounded-full px-2 py-0.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>
                          Live
                        </span>
                        <button
                          onClick={() => onResetFlow(loadedState.production_order)}
                          className="ml-2 inline-flex items-center rounded-full border border-amber-300/50 bg-amber-400/15 px-3 py-1 text-xs font-medium text-amber-100 hover:bg-amber-400/25"
                        >
                          Reset Flow
                        </button>
                      </div>
                      <h2 className="text-2xl font-bold tracking-tight">{loadedState.production_order}</h2>
                      {loadedPlan && (
                        <p className="text-sm text-slate-300 mt-1">{loadedPlan.customer} · {loadedPlan.quantity}× {loadedPlan.product}</p>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold">{progressPct}%</div>
                      <div className="text-xs text-slate-400">{completedOps} of {totalRequired} ops</div>
                    </div>
                  </div>
                  {/* Progress bar */}
                  <div className="mt-4 h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-400 to-emerald-400 rounded-full transition-all duration-500"
                      style={{ width: `${progressPct}%` }}
                    ></div>
                  </div>
                </div>

                <div className="px-4 py-3 bg-slate-900 border-t border-slate-700">
                  <div className="grid grid-cols-7 gap-2">
                    {phaseFlow.map((phase) => {
                      const finished = loadedState[phase.op]?.required && loadedState[phase.op]?.finished_at
                      const active = currentOperation === phase.op
                      return (
                        <div
                          key={phase.op}
                          className={`rounded-lg border px-2 py-1.5 text-center text-xs font-medium transition ${
                            finished
                              ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-200'
                              : active
                                ? 'bg-blue-500/25 border-blue-300/60 text-blue-100'
                                : 'bg-slate-800 border-slate-600 text-slate-300'
                          }`}
                        >
                          {phase.label}
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Order details row */}
                {loadedPlan && (
                  <div className="px-6 py-3 bg-gray-50 border-t border-gray-100 flex divide-x divide-gray-200 text-sm">
                    <div className="pr-6">
                      <span className="text-gray-400">Priority</span>
                      <span className={`ml-2 font-semibold ${loadedPlan.priority <= 2 ? 'text-red-600' : 'text-gray-800'}`}>
                        P{loadedPlan.priority} – {getPriorityLabel(loadedPlan.priority)}
                      </span>
                    </div>
                    <div className="px-6">
                      <span className="text-gray-400">Deadline</span>
                      <span className="ml-2 font-semibold text-gray-800">
                        {new Date(loadedPlan.deadline).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                      </span>
                    </div>
                    <div className="px-6">
                      <span className="text-gray-400">Scheduled</span>
                      <span className="ml-2 font-semibold text-gray-800">
                        {formatDate(loadedPlan.starts_at)} → {formatDate(loadedPlan.ends_at)}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <LiveViewInlineSection onPhaseChange={onPhaseChange} />
            </div>
          )
        })()}
      </div>
    </div>
  )
}
