import { NextResponse } from "next/server"
import { findUserByEmail } from "@/lib/db"
import { comparePassword, generateToken, COOKIE_NAME } from "@/lib/auth"

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { email, password } = body

    if (!email || !password) {
      return NextResponse.json(
        { error: "Email and password are required" },
        { status: 400 }
      )
    }

    const cleanEmail = String(email).trim().toLowerCase()
    const user = await findUserByEmail(cleanEmail)

    if (!user) {
      return NextResponse.json(
        { error: "Invalid email or password" },
        { status: 401 }
      )
    }

    const isValid = await comparePassword(password, user.password_hash)
    if (!isValid) {
      return NextResponse.json(
        { error: "Invalid email or password" },
        { status: 401 }
      )
    }

    const sessionPayload = {
      id: user.id,
      email: user.email,
      name: user.name || user.email.split("@")[0],
      avatarId: user.avatar_id || 1,
    }

    const token = generateToken(sessionPayload)

    const response = NextResponse.json({
      success: true,
      user: sessionPayload,
      token,
    })

    // Set secure HTTP-only session cookie
    response.cookies.set({
      name: COOKIE_NAME,
      value: token,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 30, // 30 days
      path: "/",
    })

    return response
  } catch (error: any) {
    console.error("[Auth Login Error]:", error)
    return NextResponse.json(
      { error: error?.message || "Failed to log in" },
      { status: 500 }
    )
  }
}
