import jwt from "jsonwebtoken"
import bcrypt from "bcryptjs"
import { cookies } from "next/headers"

const JWT_SECRET = process.env.JWT_SECRET || "satquery_neon_auth_jwt_super_secret_key_2026_isro"
const COOKIE_NAME = "satquery_session_token"

export interface UserSession {
  id: string
  email: string
  name: string
}

/**
 * Hash plain text password using bcrypt
 */
export async function hashPassword(password: string): Promise<string> {
  const salt = await bcrypt.genSalt(10)
  return bcrypt.hash(password, salt)
}

/**
 * Compare plain text password against stored hash
 */
export async function comparePassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash)
}

/**
 * Generate JWT token for user session
 */
export function generateToken(payload: UserSession): string {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: "30d" })
}

/**
 * Verify JWT token and extract user session
 */
export function verifyToken(token: string): UserSession | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as UserSession
    return {
      id: decoded.id,
      email: decoded.email,
      name: decoded.name,
    }
  } catch {
    return null
  }
}

/**
 * Get currently authenticated user from incoming request (Cookie or Authorization header)
 */
export async function getSessionUser(req?: Request): Promise<UserSession | null> {
  // 1. Check Authorization header
  if (req) {
    const authHeader = req.headers.get("Authorization")
    if (authHeader && authHeader.startsWith("Bearer ")) {
      const token = authHeader.slice(7).trim()
      const user = verifyToken(token)
      if (user) return user
    }
  }

  // 2. Check HTTP cookies
  try {
    const cookieStore = await cookies()
    const sessionCookie = cookieStore.get(COOKIE_NAME)
    if (sessionCookie && sessionCookie.value) {
      return verifyToken(sessionCookie.value)
    }
  } catch {}

  return null
}

export { COOKIE_NAME }
