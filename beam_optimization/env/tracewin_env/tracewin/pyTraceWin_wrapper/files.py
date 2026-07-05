"""
TraceWin file I/O (vendored, trimmed).
Source: ESS python tools — https://gitlab.esss.lu.se/ess-bp/ess-python-tools/-/blob/mamad/ess/TraceWin.py
Only the classes used by this project are kept: Dst (.dst particle
distributions) and Plt (.plt envelope files). The original file also contained
DensityFile, RemoteDataMerger, Partran and FieldMap, removed as unused.
"""

from __future__ import print_function


class Dst:
    """
    Simple class to read in a
    TraceWin distribution file

    Class afterwards hold the following
    dictionary items:
      - x [m]
      - xp [rad]
      - y [m]
      - yp [rad]
      - phi [rad]
      - E [MeV] (kinetic energy)
    """

    def __init__(self, filename=None):
        # easy storage..
        self.filename = filename
        # used to create dict behaviour..
        self._columns = ["x", "xp", "y", "yp", "phi", "E"]
        if filename:
            # read in the file..
            self._readBinaryFile()
        else:
            import numpy

            self.Np = 0
            self.Ib = 0.0
            self.freq = 352.21
            self._data = numpy.zeros((self.Np, 6))
            self.mass = 0.0

    def append(self, x=0.0, xp=0.0, y=0.0, yp=0.0, E=0.0, phi=0.0):
        """
        Append one particle to the distribution

        - Kinetic Energy in MeV
        - x,y in m
        - xp,yp in rad
        - phi in rad
        """
        import numpy

        self._data = numpy.append(self._data, [[x, xp, y, yp, phi, E]], 0)
        self.Np += 1

    def remove(self, i=None):
        """
        Removes all particles from the distribution, or the line specified by i
        """
        import numpy

        if i is None:
            self._data = numpy.delete(self._data, numpy.s_[:], 0)
            self.Np = 0
        else:
            self._data = numpy.delete(self._data, i, 0)
            self.Np -= 1

    def _readBinaryFile(self):
        # Thanks Ema!

        import numpy

        fin = open(self.filename, "r")

        # dummy, Np, Ib, freq, dummy
        Header_type = numpy.dtype(
            [
                ("dummy12", numpy.int16),
                ("Np", numpy.int32),
                ("Ib", numpy.float64),
                ("freq", numpy.float64),
                ("dummy3", numpy.int8),
            ]
        )
        Header = numpy.fromfile(fin, dtype=Header_type, count=1)
        self.Np = Header["Np"][0]
        self.Ib = Header["Ib"][0]
        self.freq = Header["freq"][0]

        # Some toutatis distributions has an undocumented 7th line of 0's
        Table = numpy.fromfile(fin, dtype=numpy.float64, count=self.Np * 7 + 1)
        if len(Table) == self.Np * 7 + 1:
            self._data = Table[:-1].reshape(self.Np, 7)
        elif len(Table) == self.Np * 6 + 1:  # this is true in most cases
            self._data = Table[:-1].reshape(self.Np, 6)
        else:
            raise ValueError("Incorrect table dimensions found:", len(Table))

        # convert x,y from cm to m:
        self._data[:, 0] *= 1e-2
        self._data[:, 2] *= 1e-2

        self.mass = Table[-1]

    def keys(self):
        return self._columns[:]

    def __getitem__(self, key):
        # makes the class function as a dictionary
        # e.g. dst['x'] returns the x array..
        try:
            i = self._columns.index(key)
            return self._data[:, i]
        except:
            raise ValueError("Available keys: " + str(self._columns))

    def __setitem__(self, key, value):
        try:
            i = self._columns.index(key)
            self._data[:, i] = value
        except:
            raise ValueError("Available keys: " + str(self._columns))

    def save(self, filename, toutatis=False):
        """
        Save the distribution file
        so it can be read by TraceWin again

        :param filename: Name of file
        :param toutatis: Include 7th column of zeros

        Stolen from Ryoichi's func.py (with permission)
        """

        from struct import pack
        import numpy

        fout = open(filename, "wb")
        fout.write(pack("b", 125))
        fout.write(pack("b", 100))
        fout.write(pack("i", self.Np))
        fout.write(pack("d", self.Ib))
        fout.write(pack("d", self.freq))
        fout.write(pack("b", 125))

        data = self._data.copy()

        if toutatis and data.shape[1] == 6:
            data = numpy.append(data, numpy.zeros((len(data), 1)), 1)
        elif not toutatis and data.shape[1] == 7:
            data = data[:, :-1]

        # convert x,y from m to cm:
        data[:, 0] *= 1e2
        data[:, 2] *= 1e2

        if toutatis:
            data = data.reshape(self.Np * 7, 1)
        else:
            data = data.reshape(self.Np * 6, 1)

        for x in data:
            fout.write(pack("d", x))

        fout.write(pack("d", self.mass))
        fout.close()

    def subplot(self, index, x, y=None, nb=100, mask=None):
        """
        Create a subplot histogram similar to TraceWin.

        Example::
            import numpy as np
            from ess import TraceWin
            from matplotlib import pyplot as plt
            data=TraceWin.dst('part_dtl1.dst')
            m=np.where(data['E']>3.5)
            data.subplot(221,'x','xp',mask=m)
            data.subplot(222,'y','yp',mask=m)
            data.subplot(223,'phi','E',mask=m)
            data.subplot(224,'x','y',mask=m)
            plt.show()

        """
        from matplotlib.colors import LogNorm
        import matplotlib.pyplot as plt
        import numpy as np

        units = {
            "x": "mm",
            "y": "mm",
            "xp": "mrad",
            "yp": "mrad",
            "E": "MeV",
            "phi": "deg",
        }
        # get X and Y data
        dx = np.array(self[x])
        if mask != None:
            dx = dx[mask]
        if y != None:
            dy = np.array(self[y])
            if mask != None:
                dy = dy[mask]

        if x in ["x", "y", "xp", "yp"]:
            dx *= 1e3
        if y in ["x", "y", "xp", "yp"]:
            dy *= 1e3
        if x in ["phi"]:
            dx -= np.average(dx)
            dx *= 180 / np.pi
        if y in ["phi"]:
            dy -= np.average(dy)
            dy *= 180 / np.pi
        if x in ["E"] and max(dx) < 0.1:
            dx *= 1e3
            units["E"] = "keV"
        if y in ["E"] and max(dy) < 0.1:
            dy *= 1e3
            units["E"] = "keV"

        plt.subplot(index)
        if y != None:
            plt.hist2d(dx, dy, bins=nb, norm=LogNorm())
            plt.title("{} [{}] - {} [{}]".format(x, units[x], y, units[y]))
            hist, bin_edges = np.histogram(dx, bins=nb)
            b = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
            plt.plot(
                b,
                hist * 0.2 * (max(dy) - min(dy)) / max(hist) + min(dy),
                "k",
                lw=1.5,
                drawstyle="steps",
            )
            hist, bin_edges = np.histogram(dy, bins=nb)
            b = bin_edges[:-1] + 0.0 * (bin_edges[1] - bin_edges[0])
            plt.plot(
                hist * 0.2 * (max(dx) - min(dx)) / max(hist) + min(dx),
                b,
                "k",
                lw=1.5,
                drawstyle="steps",
            )
        else:
            # plot a simple 1D histogram..
            plt.hist(dx, bins=nb)
            plt.title("{} [{}]".format(x, units[x]))


class Plt:
    """
    Simple class to read in a
    TraceWin plot file

    Class afterwards hold the following
    dictionary items:
      - Ne (number of locations)
      - Np (number of particles)
      - Ib [A] (beam current)
      - freq [MHz]
      - mc2  [MeV]
      - Nelp [m] (locations)

    each plt[i], where i is element number, holds:
      - Zgen [cm] (location)
      - phase0 [deg] (ref phase)
      - wgen [MeV] (ref energy)
      - x [array, m]
      - xp [array, rad]
      - y [array, m]
      - yp [array, rad]
      - phi [array, rad]
      - E [array, MeV]
      - l [array] (is lost)

      Example::
        plt=ess.TraceWin.plt('calc/dtl1.plt')
        for i in [97,98]:
          data=plt[i]$
          if data:
            print(data['x'])
    """

    def __init__(self, filename):
        # easy storage..
        self.filename = filename
        # used to create dict behaviour..
        self._columns = ["x", "xp", "y", "yp", "phi", "E", "l"]
        # read in the file..
        self._readBinaryFile()

    def _readBinaryFile(self):
        # Thanks Emma!

        import numpy

        fin = open(self.filename, "r")

        # dummy, Np, Ib, freq, dummy
        Header_type = numpy.dtype(
            [
                ("dummy12", numpy.int16),
                ("Ne", numpy.int32),
                ("Np", numpy.int32),
                ("Ib", numpy.float64),
                ("freq", numpy.float64),
                ("mc2", numpy.float64),
            ]
        )
        SubHeader_type = numpy.dtype(
            [
                ("dummy12", numpy.int8),
                ("Nelp", numpy.int32),
                ("Zgen", numpy.float64),
                ("phase0", numpy.float64),
                ("wgen", numpy.float64),
            ]
        )

        Header = numpy.fromfile(fin, dtype=Header_type, count=1)
        self.Np = Header["Np"][0]
        self.Ne = Header["Ne"][0]
        self.Ib = Header["Ib"][0]
        self.freq = Header["freq"][0]
        self.mc2 = Header["mc2"][0]

        self._data = []
        self.Nelp = []

        i = 0
        while i < self.Ne:
            SubHeader = numpy.fromfile(fin, dtype=SubHeader_type, count=1)
            # unfinished files need this fix (simulation still running)
            if len(SubHeader["Nelp"]) == 0:
                break
            i = SubHeader["Nelp"][0]

            self.Nelp.append(i)

            Table = numpy.fromfile(fin, dtype=numpy.float32, count=self.Np * 7)
            Table = Table.reshape(self.Np, 7)
            data = {}
            for key in ["Zgen", "phase0", "wgen"]:
                data[key] = SubHeader[key][0]
            for j in range(7):
                c = self._columns[j]
                data[c] = Table[:, j]
                # convert x,y from cm to m
                if c in ["x", "y"]:
                    data[c] *= 1e-2
            self._data.append(data)

    def __getitem__(self, key):
        if key in self.Nelp:
            import numpy

            i = self.Nelp.index(key)

            ret = {}
            # some particles are lost, exclude those:
            lost_mask = self._data[i]["l"] == 0
            for key in self._data[i]:
                if isinstance(self._data[i][key], numpy.ndarray):
                    ret[key] = self._data[i][key][lost_mask]
                else:
                    ret[key] = self._data[i][key]
            return ret
        else:
            print("No data to plot at element", key)

    def calc_s(self):
        """
        Generates self.s which holds
        the position of each element
        in metres
        """
        import numpy

        self.s = []
        for i in self.Nelp:
            self.s.append(self[i]["Zgen"] / 100.0)
        self.s = numpy.array(self.s)

    def calc_avg(self):
        """
        Calculates averages of 6D coordinates at each
        element, such that e.g.
        self.avg["x"] gives average X at each location.

        Units: m, rad, MeV
        """
        import numpy

        self.avg = dict(x=[], xp=[], y=[], yp=[], E=[], phi=[])

        vals = self._columns[:-1]

        for i in self.Nelp:
            data = self[i]
            for v in vals:
                self.avg[v].append(numpy.average(data[v]))

    def calc_rel(self):
        """
        Calculates relativistic gamma/beta
        at each position, based on
        AVERAGE beam energy
        (NOT necessarily reference)
        """
        import numpy

        if not hasattr(self, "avg"):
            self.calc_avg()
        self.gamma = []
        self.beta = []
        for i, j in zip(self.Nelp, range(len(self.Nelp))):
            Eavg = self.avg["E"][j]
            self.gamma.append((self.mc2 + Eavg) / self.mc2)
            self.beta.append(numpy.sqrt(1.0 - 1.0 / self.gamma[-1] ** 2))
        self.gamma = numpy.array(self.gamma)
        self.beta = numpy.array(self.beta)

    def calc_minmax(self, pmin=5, pmax=95):
        """
        Calculates min/max values of beam coordinates
        in percentile, pmin is lower and pmax upper.

        Units: cm
        """
        import numpy

        self.min = dict(x=[], xp=[], y=[], yp=[], E=[])
        self.max = dict(x=[], xp=[], y=[], yp=[], E=[])

        for i in self.Nelp:
            data = self[i]
            for v in self.min.keys():
                self.min[v].append(numpy.percentile(data[v], pmin))
                self.max[v].append(numpy.percentile(data[v], pmax))

        for v in self.min.keys():
            self.min[v] = numpy.array(self.min[v])
            self.max[v] = numpy.array(self.max[v])

    def calc_sigma(self):
        """
        Calculates the sigma matrix

        Creates self.sigma such that self.sigma[i,j]
        returns the sigma matrix for value i,j.

        The numbering is:
        0: x
        1: xp
        2: y
        3: yp
        4: E
        5: phi
        """

        import numpy

        if not hasattr(self, "avg"):
            self.calc_avg()

        vals = self._columns[:-1]

        self.sigma = []
        for j in range(len(self.Nelp)):
            i = self.Nelp[j]
            data = self[i]

            self.sigma.append(
                [
                    [
                        numpy.mean(
                            (data[n] - self.avg[n][j]) * (data[m] - self.avg[m][j])
                        )
                        for n in vals
                    ]
                    for m in vals
                ]
            )

        self.sigma = numpy.array(self.sigma)

    def calc_std(self):
        """
        Calculates the beam sizes

        """

        import numpy

        if not hasattr(self, "sigma"):
            self.calc_sigma()

        vals = self._columns[:-1]

        self.std = {}

        for j in range(len(vals)):
            v = vals[j]
            self.std[v] = numpy.sqrt(self.sigma[:, j, j])

    def calc_twiss(self):
        """
        Calculates emittance, beta, alfa, gamma
        for each plane, x-xp, y-yp, and E-phi
        """

        import numpy

        if not hasattr(self, "sigma"):
            self.calc_sigma()
        if not hasattr(self, "gamma"):
            self.calc_rel()

        self.twiss_eps = []
        for j in range(len(self.Nelp)):
            self.twiss_eps.append(
                [
                    numpy.sqrt(numpy.linalg.det(self.sigma[j][i : i + 2][:, i : i + 2]))
                    for i in (0, 2, 4)
                ]
            )
        self.twiss_eps = numpy.array(self.twiss_eps)

        # Calculate normalized emittance:
        # TODO: this is NOT correct normalization for longitudinal
        self.twiss_eps_normed = self.twiss_eps.copy()
        for i in range(3):
            self.twiss_eps_normed[:, i] *= self.gamma * self.beta

        # Calculate beta:
        # This is a factor 10 different from what TraceWin plots
        self.twiss_beta = [
            [self.sigma[j][i][i] / self.twiss_eps[j, i // 2] for i in (0, 2, 4)]
            for j in range(len(self.Nelp))
        ]
        self.twiss_beta = numpy.array(self.twiss_beta)

        # Calculate alpha:
        self.twiss_alpha = [
            [-self.sigma[j][i][i + 1] / self.twiss_eps[j, i // 2] for i in (0, 2, 4)]
            for j in range(len(self.Nelp))
        ]
        self.twiss_alpha = numpy.array(self.twiss_alpha)

    def get_dst(self, index):
        """
        Returns the dst corresponding to the given index
        """
        import numpy

        dset = self[index]

        _dst = Dst()
        _dst.freq = self.freq
        _dst.Ib = self.Ib * 1000
        _dst.Np = len(dset["x"])
        _dst.mass = self.mc2
        _dst._data = numpy.array(
            [dset["x"], dset["xp"], dset["y"], dset["yp"], dset["phi"], dset["E"]]
        ).transpose()
        return _dst

    def save_dst(self, index, filename):
        """
        Saves the dst at the specified index to file

        Returns the same dst object.
        """
        _dst = self.get_dst(index)
        _dst.save(filename)
        return _dst
