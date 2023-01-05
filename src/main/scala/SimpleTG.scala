import it.unimi.dsi.fastutil.longs.LongOpenHashBigSet
import org.apache.hadoop.io.LongWritable
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object SimpleTG extends Serializable {
  def main(args: Array[String]): Unit = {
    val scale = if (args.length > 0) args(0).toInt else 20
    val rootPath = if (args.length > 1) args(1) else System.currentTimeMillis().toString
    val path = s"$rootPath/simple-tg/scale$scale/${System.currentTimeMillis().toString}"
    val ratio = if (args.length > 2) args(2).toInt else 16

    val (a, b, c, d) = (0.57d, 0.19d, 0.19d, 0.05d)

    val numVertices = math.pow(2, scale).toInt
    val numEdges = ratio * numVertices

    val rng: Long = System.currentTimeMillis()

    println(s"Probabilities=($a, $b, $c, $d), |V|=$numVertices (2 ^ $scale), |E|=$numEdges ($ratio * $numVertices)")
    println(s"RandomSeed=$rng")

    val conf = new SparkConf().setAppName("Simple TG")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val startTime = System.currentTimeMillis()

    val vertexRDD = sc.range(0, numVertices - 1)
    val ds = sc.broadcast(new SKG(scale, ratio, a, b, c, d))
    val edges = vertexRDD.doRecVecGen(ds, rng)
    edges.saveAsHadoopFile(path, classOf[LongWritable], classOf[LongOpenHashBigSet], classOf[TSVOutputFormat])

    val timeSpent = System.currentTimeMillis() - startTime

    val log =
      s"""================================================================================
         |SimpleTG
         |Scale $scale
         |Generation completed in ${timeSpent / 1000d} seconds
         |================================================================================""".stripMargin
    println(log)
    sc.parallelize(log, 1).saveAsTextFile(s"$path/logs")

    sc.stop
  }

  implicit class RecVecGenClass(self: RDD[Long]) extends Serializable {
    def doRecVecGen(ds: Broadcast[_ <: SKG], rng: Long): RDD[(Long, LongOpenHashBigSet)] = {
      self.mapPartitions { partitions =>
        val skg = ds.value
        partitions.flatMap { u =>
          val random = new Random(rng + u)
          val degree = skg.getDegree(u, random)
          if (degree < 1)
            Iterator.empty
          else {
            val recVec = skg.getRecVec(u)
            val sigmas = skg.getSigmas(recVec)
            val adjacency = new LongOpenHashBigSet(degree)
            var i = 0
            while (i < degree) {
              adjacency.add(skg.determineEdge(recVec, sigmas, random))
              i += 1
            }
            Iterator((u, adjacency))
          }
        }
      }
    }
  }
}