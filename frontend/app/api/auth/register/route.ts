import { NextResponse } from "next/server"
import crypto from "crypto"
import { findUserByEmail, createUser } from "@/lib/db"
import { hashPassword, generateToken, COOKIE_NAME } from "@/lib/auth"

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { email, password, name } = body

    if (!email || !password) {
      return NextResponse.json(
        { error: "Email and password are required" },
        { status: 400 }
      )
    }

    const cleanEmail = String(email).trim().toLowerCase()
    if (!cleanEmail.includes("@") || cleanEmail.length < 5) {
      return NextResponse.json(
        { error: "Please provide a valid email address" },
        { status: 400 }
      )
    }

    if (String(password).length < 6) {
      return NextResponse.json(
        { error: "Password must be at least 6 characters long" },
        { status: 400 }
      )
    }

    // Check if user already exists
    const existing = await findUserByEmail(cleanEmail)
    if (existing) {
      return NextResponse.json(
        { error: "An account with this email already exists" },
        { status: 409 }
      )
    }

    const userId = "usr_" + crypto.randomUUID().replace(/-/g, "").slice(0, 16)
    const displayName = (name && String(name).trim()) || cleanEmail.split("@")[0]
    const passwordHash = await hashPassword(password)
    const avatarId = Number(body.avatarId) || 1

    const newUser = await createUser({
      id: userId,
      email: cleanEmail,
      name: displayName,
      passwordHash,
      avatarId,
    })

    const sessionPayload = {
      id: newUser.id,
      email: newUser.email,
      name: newUser.name,
      avatarId: newUser.avatar_id || avatarId,
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
    console.error("[Auth Register Error]:", error)
    return NextResponse.json(
      { error: error?.message || "Failed to register account" },
      { status: 500 }
    )
  }
}
