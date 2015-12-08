lazy val commonSettings = Seq(
  organization := "com.feynmanliang",
  version := "0.1.0-SNAPSHOT",
  scalaVersion := "2.11.5"
)

lazy val publishSonatypeOSSRH = Seq(
  publishMavenStyle := true,
  publishTo := {
    val nexus = "https://oss.sonatype.org/"
    if (isSnapshot.value)
      Some("snapshots" at nexus + "content/repositories/snapshots")
    else
      Some("releases"  at nexus + "service/local/staging/deploy/maven2")
  },
  publishArtifact in Test := false,
  pomIncludeRepository := { _ => false }, // remove repos for optional dependencies
  licenses := Seq("MIT" -> url("http://www.opensource.org/licenses/MIT")),
  homepage := Some(url("https://github.com/feynmanliang/optala")),
	pomExtra := (
		<scm>
			<url>git@github.com:feynmanliang/optala.git</url>
			<connection>scm:git:git@github.com:feynmanliang/optala.git</connection>
		</scm>
		<developers>
			<developer>
				<id>feynmanliang</id>
				<name>Feynman Liang</name>
				<url>http://feynmanliang.com</url>
			</developer>
		</developers>)
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(publishSonatypeOSSRH: _*).
  settings(
    name := "optala",
    libraryDependencies ++=  Seq(
      "org.scalanlp" %% "breeze" % "0.11.2",
      "org.scalanlp" %% "breeze-natives" % "0.11.2",
      "org.scalanlp" %% "breeze-viz" % "0.11.2",
      "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"
    ),
    resolvers ++= Seq(
      "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
      "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
    )
  )
