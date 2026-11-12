// Shared base URL — set VITE_API_URL in web/.env
export const API = import.meta.env.VITE_API_URL || 'http://localhost:4000'

export const SPOTS = [
  'Adyar', 'Anna Nagar', 'Tambaram', 'Velachery',   // Chennai
  'Gajuwaka', 'Maddilapalem', 'Madhurawada', 'Pendurthi', // Visakhapatnam
]

export const TODAY = new Date().toISOString().split('T')[0]

