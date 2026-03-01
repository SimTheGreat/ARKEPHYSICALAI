export const getPriorityColor = (priority) => {
  switch (priority) {
    case 1: return 'bg-red-100 text-red-800 border-red-300'
    case 2: return 'bg-orange-100 text-orange-800 border-orange-300'
    case 3: return 'bg-green-100 text-green-700 border-green-300'
    case 4: return 'bg-green-50 text-green-600 border-green-200'
    default: return 'bg-gray-100 text-gray-600 border-gray-300'
  }
}

export const getPriorityLabel = (priority) => {
  switch (priority) {
    case 1: return 'High'
    case 2: return 'Medium'
    case 3: return 'Normal'
    case 4: return 'Low'
    default: return 'Very Low'
  }
}

export const formatDate = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export const OPERATION_SEQUENCE = ['smt', 'reflow', 'tht', 'aoi', 'test', 'coating', 'pack']

export const PHASE_TO_OPERATION = {
  SMT: 'smt',
  Reflow: 'reflow',
  THT: 'tht',
  AOI: 'aoi',
  Test: 'test',
  Coating: 'coating',
  Pack: 'pack',
}
